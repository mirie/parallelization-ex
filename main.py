"""Simple example to explore parallelization in an agentic system
   Source: Agentic Design Patterns / Antonio Gulli tutorial / code sample
"""

import argparse
import os
import asyncio
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough


def build_map_chain(llm: BaseChatModel) -> Runnable:
    """Build the block of independent chains that run in parallel."""

    # --- Define independent chains ---
    # These 3 chains represent distinct tasks that can be executed in parallel.
    summarize_chain: Runnable = (
        ChatPromptTemplate.from_messages([
            ("system", "Summarize the following topic concisely:"),
            ("user", "{topic}")
        ])
        | llm
        | StrOutputParser()
    )

    questions_chain: Runnable = (
        ChatPromptTemplate.from_messages([
            ("system", "Generate three interesting questions about the following topic:"),
            ("user", "{topic}")
        ])
        | llm
        | StrOutputParser()
    )

    terms_chain: Runnable = (
        ChatPromptTemplate.from_messages([
            ("system", "Identify 5-10 key terms from the following topic, separated by commas:"),
            ("user", "{topic}")
        ])
        | llm
        | StrOutputParser()
    )

    # --- Build the parallel + synthesis chain ---
    # 1. Define the block of tasks to run in parallel. The results of these, along with the original topic, will be fed into
    # the next step.
    return RunnableParallel(
        {
            "summary": summarize_chain,
            "questions": questions_chain,
            "key_terms": terms_chain,
            "topic": RunnablePassthrough(),  # Pass the original topic through
        }
    )


def build_synthesis_chain(llm: BaseChatModel) -> Runnable:
    """Build the chain that combines the parallel results into one answer."""

    # 2. Define the final synthesis prompt which will combine the parallel results.
    synthesis_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Based on the following information:\n"
         "Summary: {summary}\n"
         "Related Questions: {questions}\n"
         "Key Terms: {key_terms}\n"
         "Synthesize a comprehensive answer."),
        ("user", "Original topic: {topic}")
    ])

    return synthesis_prompt | llm | StrOutputParser()


def build_parallel_chain(llm: BaseChatModel) -> Runnable:
    """Build the parallel + synthesis chain around the given LLM."""
    # 3. Construct the full chain by piping the parallel results directly into the synthesis chain.
    return build_map_chain(llm) | build_synthesis_chain(llm)


async def run_parallel_example(chain: Runnable, topic: str) -> None:
    """
    Asynchronously invokes the parallel processing chain with a specific topic
    and prints the synthesized result.
    Args:
        chain: The chain to run.
        topic: The input topic to be processed by the LangChain chains.
    """
    print(f"\n--- Running parallel LangChain example for topic: '{topic}' ---")
    try:
        # The input to 'ainvoke' is the single 'topic' string, then passed to each runnable
        # in the `map_chain`.
        response = await chain.ainvoke(topic)
        print("\n--- Final response ---")
        print(response)
    except Exception as e:
        print(f"\nAn error occurred during chain execution: {e}")


async def run_debug_example(map_chain: Runnable, synthesis_chain: Runnable, topic: str) -> None:
    """Run the two stages separately, printing the parallel results and then the synthesis."""
    print(f"\n--- Running parallel LangChain example (debug) for topic: '{topic}' ---")
    try:
        results = await map_chain.ainvoke(topic)
        for key, value in results.items():
            print(f"\n===== {key} =====\n{value}")

        # The same results go into the synthesis step, so the two outputs come from one run.
        response = await synthesis_chain.ainvoke(results)
        print("\n===== synthesis (final response) =====")
        print(response)
    except Exception as e:
        print(f"\nAn error occurred during chain execution: {e}")


def main() -> None:
    # --- Get topic from argument if passed, otherwise use default. ---
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "topic",
        nargs="?",
        default="The history of space exploration",
        help="Topic to process (default: %(default)s)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print each parallel chain's output before the final synthesis",
    )
    args = parser.parse_args()

    # --- Configuration ---
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY in your environment or .env file")

    model_name = os.getenv("OPENAI_MODEL", "gpt-5-mini")
    llm = ChatOpenAI(model=model_name, temperature=0.7)

    # --- Run ---
    if args.debug:
        asyncio.run(run_debug_example(build_map_chain(llm), build_synthesis_chain(llm), args.topic))
        return

    chain = build_parallel_chain(llm)
    asyncio.run(run_parallel_example(chain, args.topic))


if __name__ == "__main__":
    main()
