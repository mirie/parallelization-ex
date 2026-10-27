# parallelization-ex

Two small examples of the parallelization pattern from Antonio Gulli's *Agentic Design Patterns*:

- **[LangChain](#langchain-example)** (`langchain_parallel.py`): three chains run in parallel on a topic, then a synthesis step combines them.
- **[Google ADK](#google-adk-example)** (`adk_parallel/`): three researcher agents search the web in parallel, then a synthesis agent writes a structured report.

## LangChain example

Three independent chains (summary, three questions, key terms) run in parallel on a topic. Their outputs, plus the original topic, are then fed into a synthesis step that writes the final answer.

### Setup

```bash
cp .env.example .env   # then set OPENAI_API_KEY (and optionally OPENAI_MODEL)
uv run langchain_parallel.py
```

### Usage

```bash
uv run langchain_parallel.py                      # default topic, final answer only
uv run langchain_parallel.py "How vaccines work"  # custom topic
uv run langchain_parallel.py --debug              # show each parallel output, then the synthesis
```

`--debug` runs the parallel stage once and passes those same results into the synthesis, so you can compare the intermediate outputs with the final answer from a single run.

### Findings: the synthesis step pulls from training data

Running `--debug` shows where the final answer's content comes from. Exact outputs vary from run to run, but the pattern is consistent.

#### Nothing in this pipeline has source material

The only input is the topic string. Even the stage-1 outputs (the summary, the questions, the key terms) come from the model's training data. There is no document, search or retrieval step anywhere.

#### The synthesis adds facts that were not in the stage-1 outputs

The synthesis prompt says "Synthesize a comprehensive answer." It does not say "using only the information above", so the model treats the stage-1 outputs as an outline and fills in the rest from what it knows.

- **Reused from the inputs:** parts of the synthesis restate the stage-1 summary, lightly reworded, and the key terms appear where expected.
- **Added by the synthesis (in none of the stage-1 outputs):** extra facts, names, programs and whole sections that no stage-1 output mentioned.

#### Why

The generated questions contain no facts, only prompts. Answering them requires knowledge the inputs don't contain, so the model supplies it. The synthesis may also add unrequested extras, such as closing offers to expand further.

#### Accuracy risk

Facts that aren't in the inputs are unchecked, so the synthesis can state things that are wrong or misleading.

#### Ways to keep the synthesis grounded (not implemented)

1. **Restrict the prompt.** For example: "Using only the information below, answer each question in turn. Do not add facts that aren't present." The output gets shorter and thinner, which is the intended effect.
2. **Add retrieval.** Give the stage-1 chains real documents to work from. This is how the pattern is normally used.

## Google ADK example

Three researcher agents (renewable energy, electric vehicles, carbon capture) each search the web and summarize one key finding. A synthesis agent then combines the three summaries into a structured report, using only those summaries.

The agents are wired as a [Google ADK](https://google.github.io/adk-docs/) `Workflow` in `adk_parallel/agent.py`:

```python
Workflow(edges=[(START, (researcher_1, researcher_2, researcher_3), JoinNode(name="join"), synthesis_agent)])
```

- The tuple of researchers is the parallel fan-out: all three start at the same time.
- `JoinNode` waits for all three to finish. Without it, the synthesis agent would run once per researcher.
- Each researcher stores its summary in session state via `output_key`, and the synthesis prompt reads it back with placeholders like `{renewable_energy_result}`.

`ParallelAgent` and `SequentialAgent`, which many ADK tutorials use, are deprecated in favor of `Workflow`.

### Setup

Set both `GOOGLE_API_KEY` (for the Gemini agents) and `OPENAI_API_KEY` (for web search) in `.env`.

### Usage

Run these from the project root:

```bash
uv run adk run adk_parallel "Research recent sustainable technology advancements"   # terminal
uv run adk web                                                                       # browser UI at http://localhost:8000
```

The research topics are fixed in the agents' instructions, so the message only starts the run. ADK requires one. In `adk web`, pick `adk_parallel` from the agent dropdown. The **Trace** tab shows the three researchers starting together.

A run takes about a minute, mostly spent waiting on the web searches.

### Why search goes through OpenAI

The agents run on `gemini-3.5-flash-lite`, but the researchers search with an OpenAI-backed `web_search` function tool instead of ADK's built-in `google_search`:

- On the Gemini API free tier, Google Search grounding has no quota for Gemini 3 and newer models, and requests that include it fail with `429 RESOURCE_EXHAUSTED`. The rate-limit dashboard lists free search quota for Gemini 2 and 2.5, but new accounts can't call those models.
- `web_search` calls OpenAI's Responses API with its built-in `web_search` tool, using `OPENAI_MODEL`. Each run makes three searches, billed to your OpenAI account. The Gemini calls stay on the free tier.

`web_search` must be `async`. ADK runs synchronous tools directly on its event loop, so a blocking HTTP call would stall the other researchers until each search finished, and they would effectively run one after another.

Unlike the LangChain example, this one grounds the synthesis: the researchers work from live search results, and the synthesis prompt restricts the report to their summaries.
