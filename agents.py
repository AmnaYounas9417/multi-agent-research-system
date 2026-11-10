import os

from langchain.agents import create_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import config
from tools import scrape_url, web_search

TEMPERATURE = 0.2


def build_groq():
    key = config.get_key("GROQ_API_KEY")
    if not key:
        return None
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=TEMPERATURE,
        api_key=key,
        timeout=60,
    )


def build_gemini():
    key = config.get_key("GEMINI_API_KEY")
    if not key:
        return None
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
        temperature=TEMPERATURE,
        google_api_key=key,
    )


def build_llama():
    key = config.get_key("LLAMA_API_KEY")
    base_url = config.get_key("LLAMA_BASE_URL")
    model = config.get_key("LLAMA_MODEL")
    if not (key and base_url and model):
        return None
    from langchain_openai import ChatOpenAI

    # the Llama API is OpenAI-compatible, so ChatOpenAI works with a custom base_url
    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=key,
        temperature=TEMPERATURE,
        timeout=60,
    )


def load_models():
    # order here is the fallback order
    models = []
    for name, build in [("Groq", build_groq), ("Gemini", build_gemini), ("Llama", build_llama)]:
        try:
            model = build()
        except Exception as e:
            print(f"Skipping {name}: {e}")
            continue
        if model is not None:
            models.append((name, model))

    if not models:
        raise RuntimeError(
            "No LLM key found. Add GROQ_API_KEY, GEMINI_API_KEY or LLAMA_API_KEY to your .env file."
        )
    print("Using providers in this order:", " > ".join(name for name, _ in models))
    return models


MODELS = load_models()


class FallbackRunner:
    """Holds one runnable per provider and tries them in order until one works."""

    def __init__(self, runners):
        self.runners = runners  # list of (provider name, runnable)

    def invoke(self, inputs):
        errors = []
        for i, (name, runner) in enumerate(self.runners):
            try:
                return runner.invoke(inputs)
            except Exception as e:
                errors.append(f"{name}: {e}")
                print(f"{name} failed: {e}")
                if i + 1 < len(self.runners):
                    print(f"Trying {self.runners[i + 1][0]} instead...")
        raise RuntimeError("All providers failed:\n" + "\n".join(errors))


def build_search_agent():
    return FallbackRunner([
        (
            name,
            create_agent(
                model=model,
                tools=[web_search],
                system_prompt="You are a research assistant that searches the web for reliable sources.",
                name="search_agent",
            ),
        )
        for name, model in MODELS
    ])


def build_reader_agent():
    return FallbackRunner([
        (
            name,
            create_agent(
                model=model,
                tools=[scrape_url],
                system_prompt="You are a web content reader. Carefully extract useful information from web pages.",
                name="reader_agent",
            ),
        )
        for name, model in MODELS
    ])


writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert research writer. Write clear, structured, and insightful reports.",
    ),
    (
        "human",
        """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional.""",
    ),
])
writer_chain = FallbackRunner([
    (name, writer_prompt | model | StrOutputParser()) for name, model in MODELS
])


critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    (
        "human",
        """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
...
""",
    ),
])
critic_chain = FallbackRunner([
    (name, critic_prompt | model | StrOutputParser()) for name, model in MODELS
])
