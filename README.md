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

Running `--debug` on "The history of space exploration" showed where the final answer's content came from.

### Nothing in this pipeline has source material

The only input is the topic string. Even the stage-1 outputs (the summary, the questions, the key terms) come from the model's training data. There is no document, search or retrieval step anywhere.

### The synthesis adds facts that were not in the stage-1 outputs

The synthesis prompt says "Synthesize a comprehensive answer." It does not say "using only the information above", so the model treats the stage-1 outputs as an outline and fills in the rest from what it knows.

- **Reused from the inputs:** the "Concise timeline" section is the summary again, lightly reworded. The key terms appear where expected.
- **Added by the synthesis (in none of the stage-1 outputs):** Saturn V, lunar orbit rendezvous, heat shields, bone and muscle loss, CubeSats, miniaturization, Falcon 9, NASA's COTS and Commercial Crew programs, the Artemis Accords, space debris, planetary protection, space tourism, asteroid mining, and a "Remaining technical challenges" section.

### Why

The three generated questions contain no facts, only prompts. Answering "which breakthroughs were most critical?" requires knowledge the inputs don't contain, so the model supplies it. The synthesis also added closing offers ("If you'd like, I can…") that nothing asked for.

### Accuracy risk

Facts that aren't in the inputs are unchecked. In that run the synthesis said reusable boosters were "first pioneered in concept by the Shuttle". The Shuttle reused its orbiter and solid rocket boosters, which is different from landing and reflying a booster.

### Ways to keep the synthesis grounded (not implemented)

1. **Restrict the prompt.** For example: "Using only the information below, answer each question in turn. Do not add facts that aren't present." The output gets shorter and thinner, which is the intended effect.
2. **Add retrieval.** Give the stage-1 chains real documents to work from. This is how the pattern is normally used.
