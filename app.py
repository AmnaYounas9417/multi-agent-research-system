"""Streamlit UI. Run with: streamlit run app.py"""

import re
import time

import streamlit as st

st.set_page_config(
    page_title="Multi-Agent Research System",
    layout="wide",
)

# agents.py raises an error when no key is set, so show a message instead of a traceback
try:
    from agents import build_reader_agent, build_search_agent, critic_chain, writer_chain
except Exception as exc:
    st.error(f"Could not load agents.py: {exc}")
    st.info("Check that TAVILY_API_KEY and at least one LLM key (GROQ, GEMINI or LLAMA) are set in your .env file.")
    st.stop()


def to_text(content) -> str:
    """Some models return a list of content blocks instead of a string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(block.get("text", ""))
            else:
                parts.append(str(block))
        return "\n".join(p for p in parts if p)
    return str(content)


def parse_score(feedback: str):
    match = re.search(r"Score:\s*(\d+(?:\.\d+)?)\s*/\s*10", feedback)
    return match.group(1) if match else None


def parse_verdict(feedback: str):
    match = re.search(r"One line verdict:\s*(.+)", feedback, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else None


def run_pipeline(topic: str) -> dict:
    """Same 4 steps as pipeline.py, but showing progress in the UI."""
    state = {}
    timings = {}

    with st.status("Running research pipeline...", expanded=True) as status:
        # Step 1 - search agent
        st.write("**Step 1** - Search agent is searching the web...")
        start = time.time()
        search_agent = build_search_agent()
        search_result = search_agent.invoke({
            "messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]
        })
        state["search_results"] = to_text(search_result["messages"][-1].content)
        timings["Search"] = time.time() - start
        st.write(f"Search finished in {timings['Search']:.1f}s")

        # Step 2 - reader agent
        st.write("**Step 2** - Reader agent is scraping the best source...")
        start = time.time()
        reader_agent = build_reader_agent()
        reader_result = reader_agent.invoke({
            "messages": [(
                "user",
                f"Based on the following search results about '{topic}', "
                f"pick the most relevant URL and scrape it for deeper content.\n\n"
                f"Search Results:\n{state['search_results'][:800]}",
            )]
        })
        state["scraped_content"] = to_text(reader_result["messages"][-1].content)
        timings["Reader"] = time.time() - start
        st.write(f"Reading finished in {timings['Reader']:.1f}s")

        # Step 3 - writer chain
        st.write("**Step 3** - Writer is drafting the report...")
        start = time.time()
        research_combined = (
            f"SEARCH RESULTS : \n {state['search_results']} \n\n"
            f"DETAILED SCRAPED CONTENT : \n {state['scraped_content']}"
        )
        state["report"] = writer_chain.invoke({
            "topic": topic,
            "research": research_combined,
        })
        timings["Writer"] = time.time() - start
        st.write(f"Report drafted in {timings['Writer']:.1f}s")

        # Step 4 - critic chain
        st.write("**Step 4** - Critic is reviewing the report...")
        start = time.time()
        state["feedback"] = critic_chain.invoke({"report": state["report"]})
        timings["Critic"] = time.time() - start
        st.write(f"Review finished in {timings['Critic']:.1f}s")

        status.update(label="Pipeline complete", state="complete", expanded=False)

    state["timings"] = timings
    return state


def set_example(text: str):
    st.session_state["topic_input"] = text


with st.sidebar:
    st.header("About")
    st.write(
        "A multi-agent research system built with LangChain, Tavily and "
        "BeautifulSoup. The model is Groq, Gemini or Llama, whichever works."
    )
    st.markdown(
        """
**Pipeline**
1. Search agent (Tavily)
2. Reader agent (BeautifulSoup)
3. Writer chain
4. Critic chain
        """
    )

    st.divider()
    st.subheader("Example topics")
    examples = [
        "Latest developments in AI agents",
        "Impact of electric vehicles on the oil market",
        "Recent breakthroughs in quantum computing",
    ]
    for i, example in enumerate(examples):
        st.button(example, key=f"example_{i}", on_click=set_example, args=(example,))

    st.divider()
    if st.button("Clear results"):
        st.session_state.pop("result", None)
        st.session_state.pop("topic", None)
        st.rerun()


st.title("Multi-Agent Research System")
st.caption("Enter a topic and let the agents search, read, write and review a research report.")

topic = st.text_input(
    "Research topic",
    key="topic_input",
    placeholder="e.g. Latest developments in AI agents",
)

run_clicked = st.button("Run Research", type="primary", disabled=not topic.strip())

if run_clicked:
    try:
        st.session_state["result"] = run_pipeline(topic.strip())
        st.session_state["topic"] = topic.strip()
    except Exception as exc:
        st.session_state.pop("result", None)
        st.error(f"Something went wrong while running the pipeline: {exc}")

result = st.session_state.get("result")

if result:
    st.divider()
    st.subheader(f"Results for: {st.session_state.get('topic', '')}")

    score = parse_score(result["feedback"])
    verdict = parse_verdict(result["feedback"])

    col1, col2 = st.columns([1, 3])
    with col1:
        st.metric("Critic score", f"{score}/10" if score else "N/A")
    with col2:
        if verdict:
            st.info(f"**Verdict:** {verdict}")

    tab_report, tab_feedback, tab_search, tab_scraped = st.tabs(
        ["Final Report", "Critic Feedback", "Search Results", "Scraped Content"]
    )

    with tab_report:
        st.markdown(result["report"])
        st.download_button(
            "Download report (.md)",
            data=result["report"],
            file_name="research_report.md",
            mime="text/markdown",
        )

    with tab_feedback:
        st.markdown(result["feedback"])

    with tab_search:
        st.text(result["search_results"])

    with tab_scraped:
        st.text(result["scraped_content"])

    with st.expander("⏱Step timings"):
        for step, seconds in result["timings"].items():
            st.write(f"**{step}:** {seconds:.1f}s")
