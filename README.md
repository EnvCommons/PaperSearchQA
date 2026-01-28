# PaperSearchQA

A biomedical question-answering environment based on the PaperSearchQA dataset from HuggingFace.

## Overview

- **Dataset**: 60K biomedical QA pairs from scientific papers
- **Splits**: train (54,907), test (5,000)
- **Evaluation**: LLM-based semantic grading with gpt-5-mini
- **Tools**:
  - `web_search(query)` - Search using Tavily API, returns titles, URLs, and snippets
  - `fetch_url(url)` - Fetch full content from a specific URL
  - `submit_answer(answer)` - Submit final answer with LLM grading

## Dataset Schema

Each task contains:
- `question`: Biomedical question
- `answer`: Reference answer
- `golden_answers`: List of acceptable answer variations (backend only)
- `pmid`: PubMed ID
- `paper_title`: Source paper title
- `category`: Question category (10 biomedical domains)

## Installation

```bash
pip install -r requirements.txt
python server.py
```

## Testing

```bash
export OPENAI_API_KEY="your-key"
export TAVILY_API_KEY="your-tavily-key"
python test_agent.py
```

## Docker

```bash
docker build -t papersearchqa .
docker run -p 8080:8080 papersearchqa
```

## Features

### Web Search with Tavily
Agents can use two complementary tools for web research:
- **`web_search`** - Uses Tavily API to search and return result snippets with URLs
- **`fetch_url`** - Fetches full content from specific URLs for deeper information

This two-step approach allows agents to first find relevant sources, then read complete content as needed.

### Semantic Grading
Answers are graded using gpt-5-mini for semantic equivalence checking. The grader:
- Compares against multiple golden answers
- Considers synonyms and paraphrasing
- Handles equivalent medical terminology
- Ignores formatting differences

### Data Loading
Both train and test parquet files (18.8 MB total) are downloaded during Docker build and loaded at module import time for fast task access.

## Dataset Source

- **HuggingFace**: [jmhb/PaperSearchQA](https://huggingface.co/datasets/jmhb/PaperSearchQA)
- **Paper**: [arXiv:2601.18207](https://arxiv.org/abs/2601.18207)

## License

Dataset license follows the HuggingFace dataset licensing terms.
