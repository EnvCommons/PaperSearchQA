# PaperSearchQA

[![OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/EnvCommons/PaperSearchQA) [![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-orange)](https://huggingface.co/datasets/jmhb/PaperSearchQA)

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

[ORLv1](https://openreward.ai/orlv1.md).

## Tasks

There are two splits in this environment:

- **train**: 54,907 tasks
- **test**: 5,000 tasks

Questions span 10 biomedical categories with answers sourced from PubMed papers.

## Reward Structure

This is a multi-turn environment. Agents can use `web_search` and `fetch_url` tools to gather information, then submit via `submit_answer`. An LLM grader (gpt-5-mini) evaluates semantic equivalence against multiple golden answers, handling synonyms, paraphrasing, and equivalent medical terminology. Reward is binary: 1.0 if correct, 0.0 if incorrect.

## Data

Data consists of Parquet files (`train-00000-of-00001.parquet`, `test-00000-of-00001.parquet`) sourced from [HuggingFace jmhb/PaperSearchQA](https://huggingface.co/datasets/jmhb/PaperSearchQA). Each row contains a question, answer, golden answer variations, PubMed ID, paper title, and category. Data is stored on the OpenReward platform.

## Tools

| Tool | Description |
|------|-------------|
| `web_search` | Search the web using Tavily API. Returns titles, URLs, and snippets. |
| `fetch_url` | Fetch full content from a specific URL. |
| `submit_answer` | Submit your final answer for LLM grading. Ends the episode. |

## Time Horizon

Multi-turn. Agents can perform multiple web searches and URL fetches before submitting a final answer.

## Environment Difficulty

PaperSearchQA evaluates biomedical question answering with web research capabilities across 10 scientific domains.

## Other Environment Requirements

- OpenAI API key required for LLM-based grading. Pass via `secrets={"openai_api_key": "..."}`.
- Tavily API key required for web search. Pass via `secrets={"tavily_api_key": "..."}`.

## Safety

Agents in PaperSearchQA answer biomedical questions using web search in a standard environment. The environment does not present direct safety risks.

## Citation

```bibtex
@article{papersearchqa2025,
  title={PaperSearchQA: A Biomedical Question Answering Dataset with Web Search},
  author={jmhb},
  journal={arXiv preprint arXiv:2601.18207},
  year={2025}
}
```
