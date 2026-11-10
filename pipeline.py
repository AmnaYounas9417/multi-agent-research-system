from agents import build_reader_agent, build_search_agent, writer_chain, critic_chain


def get_text(content):
    # some models return a list of content blocks instead of a plain string
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [b.get("text", "") if isinstance(b, dict) else str(b) for b in content]
        return "\n".join(p for p in parts if p)
    return str(content)


def run_research_pipeline(topic):
    state = {}

    print("\nStep 1 - search agent is working...")
    search_agent = build_search_agent()
    search_result = search_agent.invoke({
        "messages": [("user", f"Find recent, reliable and detailed information about: {topic}")]
    })
    state["search_results"] = get_text(search_result["messages"][-1].content)
    print("\nSearch results:\n", state["search_results"])

    print("\nStep 2 - reader agent is scraping the best source...")
    reader_agent = build_reader_agent()
    reader_result = reader_agent.invoke({
        "messages": [("user",
            f"Based on the following search results about '{topic}', "
            f"pick the most relevant URL and scrape it for deeper content.\n\n"
            f"Search Results:\n{state['search_results'][:800]}"
        )]
    })
    state["scraped_content"] = get_text(reader_result["messages"][-1].content)
    print("\nScraped content:\n", state["scraped_content"])

    print("\nStep 3 - writer is drafting the report...")
    research_combined = (
        f"SEARCH RESULTS:\n{state['search_results']}\n\n"
        f"DETAILED SCRAPED CONTENT:\n{state['scraped_content']}"
    )
    state["report"] = writer_chain.invoke({"topic": topic, "research": research_combined})
    print("\nFinal report:\n", state["report"])

    print("\nStep 4 - critic is reviewing the report...")
    state["feedback"] = critic_chain.invoke({"report": state["report"]})
    print("\nCritic feedback:\n", state["feedback"])

    return state


if __name__ == "__main__":
    topic = input("\nEnter a research topic: ")
    run_research_pipeline(topic)
