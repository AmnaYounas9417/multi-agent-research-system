# Multi-Agent Research System

A small research tool where a few LLM agents work together. You enter a topic, one agent searches the web, another reads the best page, then a writer drafts a report and a critic scores it. There is a terminal version and a Streamlit UI.

This project follows the "Build a Multi-Agent AI Research System with LangChain" tutorial by Sheryians AI School on YouTube. I used it as the base and then changed a few things (listed below).

## How it works

1. Search agent: uses Tavily to find sources
2. Reader agent: scrapes the most relevant page with BeautifulSoup
3. Writer chain: writes the report (LCEL)
4. Critic chain: gives a score out of 10, strengths, things to improve and a one line verdict

## What I changed or added

- Streamlit UI (`app.py`) with step progress, tabs for each result and a download button
- Switched the model from OpenAI to Llama through Groq, because my OpenAI credits ran out
- Automatic fallback: if one provider fails, it tries the next one (Groq, then Gemini, then Llama)
- Setup for running it on Google Colab

## Files

- `agents.py`: agents, chains and the fallback logic
- `tools.py`: Tavily search tool and BeautifulSoup scraper tool
- `pipeline.py`: runs the whole thing in the terminal
- `app.py`: Streamlit UI
- `config.py`: loads API keys from `.env` or Colab Secrets
- `.env.example`: template for the keys

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and paste your keys. You need a Tavily key and at least one of Groq, Gemini or Llama. Any provider you leave as `paste_your_...` is skipped. Don't upload your `.env` file, it is already in `.gitignore`.

## Run

```bash
python pipeline.py        # terminal
streamlit run app.py      # UI
```

## Running on Colab

Upload the files, add the same key names in Colab Secrets (turn on notebook access), then:

```python
!pip install -q -r requirements.txt

from pipeline import run_research_pipeline
state = run_research_pipeline("Latest developments in AI agents")
```

## Notes

- Model names change often. If you get a "model not found" error, set `GROQ_MODEL`, `GEMINI_MODEL` or `LLAMA_MODEL` in `.env` to a current one.
- Free tiers have rate limits, so a run can fail if you search too many times in a minute.
- Some sites block scraping, in which case the reader agent returns an error message instead of page text.

## Ideas for later

- Scrape more than one source per topic
- Save past reports
- Let the critic's feedback go back to the writer for a second draft
