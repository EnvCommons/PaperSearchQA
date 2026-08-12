# PaperSearchQA

[![OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://www.openreward.ai/GeneralReasoning/PaperSearchQA) [![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-orange)](https://huggingface.co/datasets/jmhb/PaperSearchQA)

## Description

PaperSearchQA is an environment for evaluating biomedical question answering with web search capabilities. It contains 59,907 QA pairs from scientific papers across 10 biomedical domains. Agents can search the web and fetch URLs to find information before submitting answers.

## Capabilities

- Biomedical question answering
- Web search and information retrieval
- Scientific literature comprehension
- Multi-step research and answer synthesis

## Compute Requirements

Agents are given a standard environment with no sandbox or file system access.

## License

[MIT](https://opensource.org/licenses/MIT).

## Tasks

There are two splits in this environment:

- **train**: 54,907 tasks
- **test**: 5,000 tasks

Questions span 10 biomedical categories with answers sourced from PubMed papers.

## Reward Structure

This is a multi-turn environment. Agents can use `web_search` and `web_fetch` tools to gather information, then submit via `submit_answer`. An LLM grader (gpt-5-mini) evaluates semantic equivalence against multiple golden answers, handling synonyms, paraphrasing, and equivalent medical terminology. Reward is binary: 1.0 if correct, 0.0 if incorrect.

## Data

Data consists of Parquet files (`train-00000-of-00001.parquet`, `test-00000-of-00001.parquet`) sourced from [HuggingFace jmhb/PaperSearchQA](https://huggingface.co/datasets/jmhb/PaperSearchQA). Each row contains a question, answer, golden answer variations, PubMed ID, paper title, and category. Data is stored on the OpenReward platform.

## Tools

| Tool | Description |
|------|-------------|
| `web_search` | Search the web. Returns titles, URLs, and snippets. |
| `web_fetch` | Fetch full content from a specific URL. |
| `submit_answer` | Submit your final answer for LLM grading. Ends the episode. |

`web_search` and `web_fetch` come from the OpenReward SDK's `WebToolset`, so the provider is configuration on the environment server rather than code here: unset (the default) uses `backsearch`, GR's backdated corpus; `OPENREWARD_SEARCH_BACKEND=tavily` routes to Tavily's live web instead.

## Time Horizon

Multi-turn. Agents can perform multiple web searches and URL fetches before submitting a final answer.

## Environment Difficulty

[Put env difficulty statistics here]

## Other Environment Requirements

- OpenAI API key required for LLM-based grading. Pass via `secrets={"openai_api_key": "..."}`.
- Search credentials depend on the configured backend: `api_key` for the default backsearch backend, or `tavily_api_key` when the server runs with `OPENREWARD_SEARCH_BACKEND=tavily`. Both fall back to the server process environment (`OPENREWARD_API_KEY` / `TAVILY_API_KEY`).

## Safety

Agents in PaperSearchQA answer biomedical questions using web search in a standard environment. The environment does not present direct safety risks.

## Citation

```bibtex
@misc{burgess2026papersearchqa,
  title={PaperSearchQA: Learning to Search and Reason over Scientific Papers with RLVR},
  author={James Burgess and Jan N. Hansen and Duo Peng and Yuhui Zhang and Alejandro Lozano and Min Woo Sun and Emma Lundberg and Serena Yeung-Levy},
  year={2026},
  eprint={2601.18207},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2601.18207}
}
```
