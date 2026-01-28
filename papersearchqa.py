from __future__ import annotations

import pandas as pd
import openai
from pydantic import BaseModel
from tavily import AsyncTavilyClient

from openreward.environments import Environment, JSONObject, TextBlock, ToolOutput, tool


# ============= Data Loading (module-level) =============
# Load BOTH train and test parquet files (separate downloads)

import os
from pathlib import Path

if Path("/orwd_data/").exists():
    DATA_PATH = Path("/orwd_data")
else:
    DATA_PATH = Path(__file__).parent

train_df = pd.read_parquet(DATA_PATH / "train-00000-of-00001.parquet")
test_df = pd.read_parquet(DATA_PATH / "test-00000-of-00001.parquet")

# Create task specs (public) and answers (backend only)
TASKS_BY_SPLIT = {"train": [], "test": []}
ANSWERS = {}  # Backend storage for golden_answers

for split_name, split_df in [("train", train_df), ("test", test_df)]:
    for idx, row in split_df.iterrows():
        task_id = f"papersearchqa_{split_name}_{idx}"

        # Public task spec (no golden_answers)
        TASKS_BY_SPLIT[split_name].append({
            "id": task_id,
            "question": row["question"],
            "pmid": row.get("pmid", ""),
            "paper_title": row.get("paper_title", ""),
            "category": row.get("cat", "")
        })

        # Private answer storage (convert numpy array to Python list)
        golden_answers = row["golden_answers"]
        if hasattr(golden_answers, 'tolist'):
            golden_answers = golden_answers.tolist()

        ANSWERS[task_id] = {
            "answer": row["answer"],
            "golden_answers": golden_answers  # List of acceptable answers
        }


# ============= Pydantic Models for Tool Inputs =============
class WebSearchInput(BaseModel):
    query: str


class FetchUrlInput(BaseModel):
    url: str


class SubmitAnswerInput(BaseModel):
    answer: str


# ============= Environment Class =============
class PaperSearchQA(Environment):
    """
    PaperSearchQA: A biomedical question-answering environment with optional web search
    and LLM-based semantic grading using gpt-4.1.
    """

    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        super().__init__(task_spec)

        # Extract task info
        self.task_id = str(task_spec["id"])
        self.question = str(task_spec["question"])
        self.pmid = str(task_spec.get("pmid", ""))
        self.paper_title = str(task_spec.get("paper_title", ""))
        self.category = str(task_spec.get("category", ""))

        # CRITICAL: Validate API keys from secrets (no env var fallback)
        openai_api_key = secrets.get("openai_api_key")
        if not openai_api_key:
            raise ValueError(
                "OpenAI API key required in secrets parameter. "
                "Pass secrets={'openai_api_key': 'your-key'} when creating session."
            )

        tavily_api_key = secrets.get("tavily_api_key")
        if not tavily_api_key:
            raise ValueError(
                "Tavily API key required in secrets parameter. "
                "Pass secrets={'tavily_api_key': 'your-key'} when creating session."
            )

        self.openai_client = openai.AsyncClient(api_key=openai_api_key)
        self.tavily_client = AsyncTavilyClient(api_key=tavily_api_key)

        # Load golden answers from backend storage
        answer_data = ANSWERS.get(self.task_id)
        if not answer_data:
            raise ValueError(f"Task {self.task_id} not found in dataset")

        self.answer = str(answer_data["answer"])
        self.golden_answers = answer_data["golden_answers"]

    @classmethod
    def list_splits(cls) -> list[str]:
        """Return available splits"""
        return ["train", "test"]

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        """Return task specifications for a given split (without golden_answers)"""
        if split not in TASKS_BY_SPLIT:
            raise ValueError(
                f"Unknown split: {split}. Available splits: {list(TASKS_BY_SPLIT.keys())}"
            )
        return TASKS_BY_SPLIT[split]

    async def get_prompt(self) -> list[TextBlock]:
        """Return the prompt shown to the agent"""
        prompt_text = f"""Answer the following biomedical question. You may use the web_search tool to find relevant information, and fetch_url to read specific pages if needed.

Question: {self.question}

When you have your answer, submit it using the submit_answer tool."""

        return [TextBlock(text=prompt_text)]

    @tool
    async def web_search(self, params: WebSearchInput) -> ToolOutput:
        """
        Search the web using Tavily. Returns search results with titles, URLs, and snippets.
        Use fetch_url tool to get full content from specific URLs if needed.
        """
        try:
            # Use Tavily search API
            response = await self.tavily_client.search(
                query=params.query,
                search_depth="basic",
                max_results=5
            )

            # Format results
            results = response.get("results", [])
            if not results:
                return ToolOutput(
                    blocks=[TextBlock(text="No search results found.")],
                    metadata={"query": params.query, "results": []},
                    reward=0.0,
                    finished=False
                )

            # Build display text
            display_parts = [f"Search results for: {params.query}\n"]
            for i, result in enumerate(results, 1):
                title = result.get("title", "No title")
                url = result.get("url", "")
                snippet = result.get("content", "")
                display_parts.append(f"{i}. {title}\n   URL: {url}\n   {snippet}\n")

            display_text = "\n".join(display_parts)

            return ToolOutput(
                blocks=[TextBlock(text=display_text)],
                metadata={
                    "query": params.query,
                    "results": results,
                    "count": len(results)
                },
                reward=0.0,
                finished=False
            )
        except Exception as e:
            return ToolOutput(
                blocks=[TextBlock(text=f"Web search failed: {str(e)}")],
                metadata={"query": params.query, "error": str(e)},
                reward=0.0,
                finished=False
            )

    @tool
    async def fetch_url(self, params: FetchUrlInput) -> ToolOutput:
        """
        Fetch and return the full text content from a specific URL using Tavily's extract method.
        Use this after web_search to get complete information from a page.
        """
        try:
            # Use Tavily's extract method
            response = await self.tavily_client.extract(urls=[params.url])

            # Get the extracted content
            results = response.get("results", [])
            if not results:
                return ToolOutput(
                    blocks=[TextBlock(text=f"No content extracted from {params.url}")],
                    metadata={"url": params.url, "results": []},
                    reward=0.0,
                    finished=False
                )

            # Get the first result (we only passed one URL)
            result = results[0]
            raw_content = result.get("raw_content", "")

            # Truncate if too long
            max_length = 8000
            if len(raw_content) > max_length:
                raw_content = raw_content[:max_length] + "...\n[Content truncated]"

            return ToolOutput(
                blocks=[TextBlock(text=f"Content from {params.url}:\n\n{raw_content}")],
                metadata={
                    "url": params.url,
                    "length": len(raw_content)
                },
                reward=0.0,
                finished=False
            )
        except Exception as e:
            return ToolOutput(
                blocks=[TextBlock(text=f"Failed to fetch URL: {str(e)}")],
                metadata={"url": params.url, "error": str(e)},
                reward=0.0,
                finished=False
            )

    @tool
    async def submit_answer(self, params: SubmitAnswerInput) -> ToolOutput:
        """
        Submit your final answer to the biomedical question.
        This tool will grade your answer against the golden answers and end the episode.
        """
        grader_result = await self._grade_answer(params.answer)

        reward = grader_result["reward"]
        is_correct = grader_result["is_correct"]
        justification = grader_result["justification"]

        # Format display
        result_emoji = "✅" if is_correct else "❌"
        result_text = "CORRECT" if is_correct else "INCORRECT"

        display_text = f"""{result_emoji} {result_text}

Evaluation:
{justification}

Reference Answer: {self.answer}
"""

        return ToolOutput(
            blocks=[TextBlock(text=display_text)],
            metadata={
                "task_id": self.task_id,
                "submitted_answer": params.answer,
                "golden_answers": self.golden_answers,
                "is_correct": is_correct,
                "justification": justification,
                "category": self.category
            },
            reward=reward,
            finished=True
        )

    async def _grade_answer(self, predicted_answer: str) -> dict:
        """
        Grade the answer using gpt-5-mini LLM grader.
        Compares submitted answer against golden_answers for semantic equivalence.
        Returns: {is_correct: bool, justification: str, reward: float}
        """
        # Handle empty answers
        if not predicted_answer or len(predicted_answer.strip()) == 0:
            return {
                "is_correct": False,
                "justification": "Empty or whitespace-only answer provided.",
                "reward": 0.0
            }

        # Build grader prompt
        golden_answers_text = "\n".join(f"- {ans}" for ans in self.golden_answers)

        grader_prompt = f"""You are an expert biomedical evaluator. Determine if the predicted answer is semantically equivalent to any of the golden answers.

Question: {self.question}

Golden Answers (any one is acceptable):
{golden_answers_text}

Predicted Answer: {predicted_answer}

Instructions:
1. Check if the predicted answer is semantically equivalent to ANY of the golden answers
2. Consider synonyms, paraphrasing, and equivalent medical terminology
3. Ignore minor formatting differences
4. Do NOT require exact word-for-word matches
5. Provide a brief justification (2-3 sentences)
6. End your response with EXACTLY one of these labels on a new line:
   - "CORRECT" if semantically equivalent
   - "INCORRECT" if not equivalent

Format:
[Your justification here]

CORRECT or INCORRECT"""

        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-5-mini",
                messages=[{"role": "user", "content": grader_prompt}],
            )

            grading_response = response.choices[0].message.content or ""

            # Parse CORRECT/INCORRECT (academicbrowse pattern)
            upper_response = grading_response.upper()
            is_correct = "CORRECT" in upper_response and "INCORRECT" not in upper_response

            reward = 1.0 if is_correct else 0.0

            return {
                "is_correct": is_correct,
                "justification": grading_response,
                "reward": reward
            }
        except Exception as e:
            # Fallback for API errors
            return {
                "is_correct": False,
                "justification": f"Grading failed due to error: {str(e)}",
                "reward": 0.0
            }
