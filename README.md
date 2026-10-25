# parallelization-ex

A small LangChain example of the parallelization pattern from Antonio Gulli's *Agentic Design Patterns*.

Three independent chains (summary, three questions, key terms) run in parallel on a topic. Their outputs, plus the original topic, are then fed into a synthesis step that writes the final answer.

## Setup

```bash
cp .env.example .env   # then set OPENAI_API_KEY (and optionally OPENAI_MODEL)
uv run main.py
```

## Usage

```bash
uv run main.py                      # default topic, final answer only
uv run main.py "How vaccines work"  # custom topic
uv run main.py --debug              # show each parallel output, then the synthesis
```

`--debug` runs the parallel stage once and passes those same results into the synthesis, so you can compare the intermediate outputs with the final answer from a single run.

## Findings: the synthesis step pulls from training data

Running `--debug` shows where the final answer's content comes from. Exact outputs vary from run to run, but the pattern is consistent.

### Nothing in this pipeline has source material

The only input is the topic string. Even the stage-1 outputs (the summary, the questions, the key terms) come from the model's training data. There is no document, search or retrieval step anywhere.

### The synthesis adds facts that were not in the stage-1 outputs

The synthesis prompt says "Synthesize a comprehensive answer." It does not say "using only the information above", so the model treats the stage-1 outputs as an outline and fills in the rest from what it knows.

- **Reused from the inputs:** parts of the synthesis restate the stage-1 summary, lightly reworded, and the key terms appear where expected.
- **Added by the synthesis (in none of the stage-1 outputs):** extra facts, names, programs and whole sections that no stage-1 output mentioned.

### Why

The generated questions contain no facts, only prompts. Answering them requires knowledge the inputs don't contain, so the model supplies it. The synthesis may also add unrequested extras, such as closing offers to expand further.

### Accuracy risk

Facts that aren't in the inputs are unchecked, so the synthesis can state things that are wrong or misleading.

### Ways to keep the synthesis grounded (not implemented)

1. **Restrict the prompt.** For example: "Using only the information below, answer each question in turn. Do not add facts that aren't present." The output gets shorter and thinner, which is the intended effect.
2. **Add retrieval.** Give the stage-1 chains real documents to work from. This is how the pattern is normally used.
