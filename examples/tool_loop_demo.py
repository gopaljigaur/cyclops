"""Tool loop demo — multi-step agentic tool chaining

Demonstrates the agentic loop where the LLM decides which tools to call,
in what order, and how many times, until it can produce a final answer.

Scenario: A research agent that must:
  1. search_papers()    — find relevant academic papers on a topic
  2. fetch_abstract()   — retrieve the abstract for a specific paper
  3. count_words()      — count words in a piece of text
  4. compute_stats()    — calculate mean, min, max of a list of numbers
  5. format_report()    — assemble results into a formatted summary

The agent is given a high-level goal and must chain these tools itself.
max_iterations is set so the loop does not run forever.
"""

import asyncio
import json
import math
import re

from cyclops import Agent, AgentConfig
from cyclops.toolkit import tool

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

# Default: Ollama (free, local — https://ollama.ai, then `ollama pull qwen3:4b`)
MODEL = "ollama/qwen3:4b"

# For best multi-tool chaining, a stronger model is recommended:
#   MODEL = "gpt-4o-mini"             # OPENAI_API_KEY (very reliable tool chaining)
#   MODEL = "groq/llama-3.1-8b-instant"  # GROQ_API_KEY (fast, free tier)
#   MODEL = "claude-3-haiku-20240307"    # ANTHROPIC_API_KEY


# ---------------------------------------------------------------------------
# Tool 1: search_papers
# ---------------------------------------------------------------------------

# Simulated paper database
_PAPERS = {
    "transformer": [
        {
            "id": "paper-001",
            "title": "Attention Is All You Need",
            "year": 2017,
            "citations": 85000,
        },
        {
            "id": "paper-002",
            "title": "BERT: Pre-training of Deep Bidirectional Transformers",
            "year": 2018,
            "citations": 62000,
        },
        {
            "id": "paper-003",
            "title": "GPT-3: Language Models are Few-Shot Learners",
            "year": 2020,
            "citations": 28000,
        },
    ],
    "reinforcement learning": [
        {
            "id": "paper-004",
            "title": "Playing Atari with Deep Reinforcement Learning",
            "year": 2013,
            "citations": 14000,
        },
        {
            "id": "paper-005",
            "title": "Proximal Policy Optimization Algorithms",
            "year": 2017,
            "citations": 12000,
        },
        {
            "id": "paper-006",
            "title": "AlphaGo: Mastering the game of Go",
            "year": 2016,
            "citations": 9500,
        },
    ],
    "diffusion": [
        {
            "id": "paper-007",
            "title": "Denoising Diffusion Probabilistic Models",
            "year": 2020,
            "citations": 18000,
        },
        {
            "id": "paper-008",
            "title": "High-Resolution Image Synthesis with Latent Diffusion Models",
            "year": 2022,
            "citations": 11000,
        },
    ],
}


@tool
def search_papers(topic: str, max_results: int = 3) -> str:
    """Search for academic papers on a given topic. Returns paper IDs, titles, years, and citation counts."""
    topic_lower = topic.lower()
    results = []
    for key, papers in _PAPERS.items():
        if key in topic_lower or topic_lower in key:
            results.extend(papers[:max_results])
            break

    if not results:
        # Fuzzy fallback
        for key, papers in _PAPERS.items():
            for word in topic_lower.split():
                if word in key:
                    results.extend(papers[:max_results])
                    break
            if results:
                break

    if not results:
        return json.dumps({"error": f"No papers found for topic: {topic}"})

    return json.dumps(results[:max_results], indent=2)


# ---------------------------------------------------------------------------
# Tool 2: fetch_abstract
# ---------------------------------------------------------------------------

_ABSTRACTS = {
    "paper-001": (
        "The dominant sequence transduction models are based on complex recurrent or convolutional neural "
        "networks that include an encoder and a decoder. The best performing models also connect the encoder "
        "and decoder through an attention mechanism. We propose a new simple network architecture, the "
        "Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions "
        "entirely. Experiments on two machine translation tasks show these models to be superior in quality "
        "while being more parallelizable and requiring significantly less time to train."
    ),
    "paper-002": (
        "We introduce a new language representation model called BERT, which stands for Bidirectional Encoder "
        "Representations from Transformers. Unlike recent language representation models, BERT is designed to "
        "pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left "
        "and right context in all layers. As a result, the pre-trained BERT model can be fine-tuned with just "
        "one additional output layer to create state-of-the-art models for a wide range of tasks."
    ),
    "paper-007": (
        "We present high quality image synthesis results using diffusion probabilistic models, a class of "
        "latent variable models inspired by considerations from nonequilibrium thermodynamics. Our best results "
        "are obtained by training on a weighted variational bound designed according to a novel connection "
        "between diffusion probabilistic models and denoising score matching with Langevin dynamics, and our "
        "models naturally admit a progressive lossy decompression scheme that can be interpreted as a "
        "generalization of autoregressive decoding."
    ),
}


@tool
def fetch_abstract(paper_id: str) -> str:
    """Fetch the abstract text for a paper given its ID."""
    abstract = _ABSTRACTS.get(paper_id)
    if abstract:
        return json.dumps({"paper_id": paper_id, "abstract": abstract})
    return json.dumps({"error": f"Abstract not found for paper_id: {paper_id}"})


# ---------------------------------------------------------------------------
# Tool 3: count_words
# ---------------------------------------------------------------------------


@tool
def count_words(text: str) -> str:
    """Count the number of words in a piece of text."""
    words = re.findall(r"\b\w+\b", text)
    return json.dumps({"word_count": len(words), "character_count": len(text)})


# ---------------------------------------------------------------------------
# Tool 4: compute_stats
# ---------------------------------------------------------------------------


@tool
def compute_stats(numbers: str) -> str:
    """Compute mean, min, max, and total of a comma-separated list of numbers."""
    try:
        values = [float(x.strip()) for x in numbers.split(",") if x.strip()]
        if not values:
            return json.dumps({"error": "No numbers provided"})
        return json.dumps(
            {
                "count": len(values),
                "total": sum(values),
                "mean": round(sum(values) / len(values), 2),
                "min": min(values),
                "max": max(values),
                "std_dev": round(
                    math.sqrt(
                        sum((x - sum(values) / len(values)) ** 2 for x in values)
                        / len(values)
                    ),
                    2,
                ),
            }
        )
    except ValueError as exc:
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# Tool 5: format_report
# ---------------------------------------------------------------------------


@tool
def format_report(title: str, findings: str, conclusion: str) -> str:
    """Format a structured research report from collected findings.

    title      : Report headline
    findings   : Bullet-point findings (use '|' to separate bullets)
    conclusion : One-paragraph concluding summary
    """
    bullets = [f"  - {f.strip()}" for f in findings.split("|") if f.strip()]
    report = (
        f"\n{'=' * 60}\n"
        f"RESEARCH REPORT: {title}\n"
        f"{'=' * 60}\n\n"
        f"KEY FINDINGS:\n" + "\n".join(bullets) + f"\n\nCONCLUSION:\n  {conclusion}\n"
        f"{'=' * 60}\n"
    )
    return report


# ---------------------------------------------------------------------------
# Sync demo
# ---------------------------------------------------------------------------


def demo_sync_tool_loop() -> None:
    print("=" * 60)
    print("Synchronous tool loop demo")
    print("=" * 60)

    config = AgentConfig(
        model=MODEL,
        system_prompt=(
            "You are a research assistant. To answer questions you MUST use the tools "
            "provided in sequence: first search for papers, then fetch abstracts for the "
            "most relevant ones, count words in the abstracts, compute citation statistics, "
            "and finally format a report with your findings. Always use format_report as "
            "your last tool call before giving your final answer."
        ),
        temperature=0.2,
        max_iterations=12,  # enough headroom for a multi-step chain
    )

    all_tools = [
        search_papers,
        fetch_abstract,
        count_words,
        compute_stats,
        format_report,
    ]
    agent = Agent(config, tools=all_tools)

    task = (
        "Research the topic of transformer models. Find the top papers, get their abstracts, "
        "analyse the citation counts statistically, and produce a formatted research report."
    )

    print(f"\nTask: {task}\n")
    print("Running agent (watch it chain tools automatically)...\n")

    result = agent.run(task)

    print("Final answer:")
    print(result)

    # Show which tools were called (visible in history)
    tool_turns = [msg for msg in agent.messages if msg.get("role") == "tool"]
    print(f"\nTotal tool calls made: {len(tool_turns)}")
    for t in tool_turns:
        print(f"  -> {t.get('name', 'unknown')}")


# ---------------------------------------------------------------------------
# Async demo
# ---------------------------------------------------------------------------


async def demo_async_tool_loop() -> None:
    print("\n" + "=" * 60)
    print("Asynchronous tool loop demo")
    print("=" * 60)

    config = AgentConfig(
        model=MODEL,
        system_prompt=(
            "You are a research assistant. Use the available tools to answer questions. "
            "Use search_papers first, then fetch_abstract for interesting papers, then "
            "compute_stats on the citation data, and end with format_report."
        ),
        temperature=0.2,
        max_iterations=10,
    )

    all_tools = [
        search_papers,
        fetch_abstract,
        count_words,
        compute_stats,
        format_report,
    ]
    agent = Agent(config, tools=all_tools)

    task = (
        "Look into diffusion model papers. Fetch an abstract, then compile a brief report "
        "with citation statistics."
    )

    print(f"\nTask: {task}\n")
    print("Running async agent...\n")

    result = await agent.arun(task)

    print("Final answer:")
    print(result)

    tool_turns = [msg for msg in agent.messages if msg.get("role") == "tool"]
    print(f"\nTotal tool calls made: {len(tool_turns)}")
    for t in tool_turns:
        print(f"  -> {t.get('name', 'unknown')}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo_sync_tool_loop()
    asyncio.run(demo_async_tool_loop())
