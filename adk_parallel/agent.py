import os

from google.adk.agents import LlmAgent
from google.adk.workflow import Workflow, JoinNode, START
from openai import AsyncOpenAI

MODEL = "gemini-3.5-flash-lite"


# Google Search grounding has no free quota for Gemini 3+, so search goes through
# OpenAI's web_search tool instead. ADK exposes this function to the agents as a tool.
# It must be async: ADK runs sync tools on the event loop, which would block the other
# parallel researchers until each search finished.
async def web_search(query: str) -> str:
    """Searches the web and returns a short summary of current results."""
    response = await AsyncOpenAI().responses.create(
        model=os.environ["OPENAI_MODEL"],
        tools=[{"type": "web_search"}],
        input=query,
    )
    return response.output_text


# --- 1. Define researcher sub-agents to run in parallel ---
# Researcher 1: Renewable energy
researcher_agent_1 = LlmAgent(
    name="RenewableEnergyResearcher",
    model=MODEL,
    instruction="You are an AI research assistant specializing in energy.\n"
    "Research the latest advancements in 'renewable energy sources'. "
    "Use the web_search tool provided. Summarize your key finding concisely (1-2 sentences).\n"
    "Output *only* the summary.",
    description="Researches renewable energy sources.",
    tools=[web_search],
    # Store result in state for the synthesis agent
    output_key="renewable_energy_result"
)

# Researcher 2: Electric vehicles
researcher_agent_2 = LlmAgent(
    name="EVResearcher",
    model=MODEL,
    instruction="You are an AI research assistant specializing in transportation.\n"
    "Research the latest developments in 'electric vehicle technology'. "
    "Use the web_search tool provided. Summarize your key finding concisely (1-2 sentences).\n"
    "Output *only* the summary.",
    description="Researches electric vehicle technology.",
    tools=[web_search],
    # Store result in state for the synthesis agent
    output_key="ev_technology_result"
)

# Researcher 3: Carbon capture
researcher_agent_3 = LlmAgent(
    name="CarbonCaptureResearcher",
    model=MODEL,
    instruction="You are an AI research assistant specializing in climate solutions.\n"
    "Research the current state of 'carbon capture methods'. "
    "Use the web_search tool provided. Summarize your key finding concisely (1-2 sentences).\n"
    "Output *only* the summary.",
    description="Researches carbon capture methods.",
    tools=[web_search],
    # Store result in state for the synthesis agent
    output_key="carbon_capture_result"
)

# --- 2. Define the SynthesisAgent (runs after the concurrent researcher agents) ---
# This agent takes the results stored in the session state by the parallel agents and
# synthesizes them into a single, structured response with attributions.
synthesis_agent = LlmAgent(
    name="SynthesisAgent",
    model=MODEL,  # potentially need a strong model for synthesis...
    instruction="You are an AI assistant responsible for combining research findings into a "
    "structured report. Your primary task is to synthesize the following research summaries, "
    "clearly attributing findings to their source areas. Structure your response using headings "
    "for each topic. Ensure the report is coherent and integrates the key points smoothly.\n\n"
    "**Crucially: Your entire response MUST be grounded *exclusively* on the information provided "
    "in the 'Input Summaries' below. Do NOT add any external knowledge, facts, or details not "
    "present in these specific summaries.**\n\n"
    "**Input Summaries:**\n"
    "  **Renewable Energy:**\n"
    "  {renewable_energy_result}\n"
    "  **Electric Vehicles:**\n"
    "  {ev_technology_result}\n"
    "  **Carbon Capture:**\n"
    "  {carbon_capture_result}\n\n"
    "**Output Format:**\n"
    "## Summary of Recent Sustainable Technology Advancements\n"
    "### Renewable Energy Findings\n"
    "(Based on RenewableEnergyResearcher's findings)\n"
    "[Synthesize and elaborate *only* on the renewable energy input summary provided above.]\n"
    "### Electric Vehicle Findings\n"
    "(Based on EVResearcher's findings)\n"
    "[Synthesize and elaborate *only* on the EV input summary provided above.]\n"
    "### Carbon Capture Findings\n"
    "(Based on CarbonCaptureResearcher's findings)\n"
    "[Synthesize and elaborate *only* on the carbon capture input summary provided above.]\n"
    "### Overall Conclusion\n"
    "[Provide a brief (1-2 sentences) concluding statement that connects *only* the findings "
    "presented above.]\n\n"
    "Output *only* the structured report following this format. Do not include introductory or "
    "concluding phrases outside of this structure, and strictly adhere to using only the provided "
    "input summary content.",
    description="Combines research findings from parallel agents into a structured, cited report, "
    "strictly grounded on provided inputs.",
)

# --- 3. Wire the workflow: fan out to the researchers, join, then synthesize ---
root_agent = Workflow(
    name="adk_parallel",
    edges=[
        (START,
         (researcher_agent_1, researcher_agent_2, researcher_agent_3),
         JoinNode(name="join"),
         synthesis_agent),
    ],
)
