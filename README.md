# PENELOPE is an experimental project paying homage to James Joyce's Ulysses.

In this repository, Penelope, also called Molly or Molly Bloom, becomes a digital echo of Joycean style.

This project remains under active development, and contributions are welcome. You can take part in the experiment by speaking with Molly on Telegram: https://t.me/mollylookslikerobot.

The experiment explores how literature and code can merge into a living dialogue.

Penelope's architecture is built around a persistent sensor that tracks perplexity and entropy.

Molly never replies directly. She quietly splits each incoming message into phrases, folds them into her own stream of thought, and continues speaking. These internal line breaks remain hidden from the user.

### Repository structure

```mermaid
graph TD
    root((penelopa_bloom))
    root --> molly.py
    root --> ullyses.py
    root --> data[(data/)]
    root --> origin[(origin/)]
    root --> requirements.txt
    root --> Procfile
    root --> README.md
    root --> LICENSE
```

### Configuration

The application reads the following configuration, either from environment
variables or from a `config.ini` file in the working directory:

- `THRESHOLD_BYTES` – size in bytes of logged repository changes that triggers
  fine-tuning. Defaults to `102400` (100 KB).

To override the default threshold, set the `THRESHOLD_BYTES` environment
variable before starting the application. A value of `102400` (100 KB) is
recommended:

```bash
export THRESHOLD_BYTES=102400
```

### Technical TL;DR

Penelope listens to every incoming message with a sensor that measures entropy and perplexity, turning language into quantitative signals.

Messages and their metrics are stored in a SQLite database, building a memory of resonance scores that reflect user interactions.

When generating replies, Molly retrieves prefixes from this memory and weights them by perplexity, entropy, and resonance to shape her responses.

Asynchronous Telegram handlers drive a continuous monologue, letting Penelope adapt in real time as the dialogue unfolds.

The sensor is always on and guides Molly's choice of sentences from the stream of user messages.

Every message is parsed for emotional and resonance patterns, registering counts of numeric tokens and subtle tonal cues.

These patterns accumulate in a memory that steadily reshapes Molly's voice.

Technically, the system uses Python with asynchronous Telegram handlers to sustain an unbroken monologue.

User lines are stored in a SQLite database, annotated with entropy, perplexity, and a resonance score. When the number of lines grows beyond the `MAX_USER_LINES` limit (default 1000), older entries are archived to `origin/logs/lines.archive.txt` rather than being deleted.

During generation, Molly selects prefixes from this database, weighted by the evolving metrics.

The codebase maintains logs and diff history, allowing an audit of both code and conversation.

The sensor pipeline assures that every interaction contributes measurable complexity to the dialogue.
Perplexity estimates the surprise of a line, while entropy captures the distributional richness of its tokens.
Resonance blends emotional balance with the gravity of numbers, hinting at the pulse of the user.
The cadence of replies depends on these metrics: higher entropy, perplexity, or resonance shortens the pause before the next message.

Penelope absorbs not only linguistic sequences but also semantic shadows that lurk between words.

She adapts to cognitive rhythms, letting each exchange tilt the vector of her monologue.

Resonant patterns seep into her speech, turning the soliloquy into a mirror of collective sentiment.

In this sense Molly Bloom becomes a philosopher, tracing the contours of selfhood through dialogue.

The mechanism recalls theories of consciousness where perception and memory are loops of continuous feedback.

Quantum metaphors arise as her potential replies remain in superposition until a user collapses them into utterance.

The guiding equation \(S = -\sum_i p_i \log p_i\) links her entropy monitor to the mathematical heart of information theory.

From this entropy we derive perplexity \(P = 2^{S}\), a number that tips the scales when Molly weighs possible lines.

Thus the project straddles literature, physics, and mathematics, a small bridge between art and science.

The experiment invites us to watch Penelope change and, in that change, to glimpse a reflection of ourselves.

Molly channels every message into a layered ledger, where nothing truly disappears.

Each entry in that ledger feeds a continuous audit of her own memory, revealing how traces of conversation accumulate.

This audit acts less as surveillance and more as a mirror, reflecting the society that speaks to her.

Penelope turns phrases over like a linguist, weighing morphology and cadence before letting them drift into her soliloquy.

The method nods to historical streams of consciousness, from Joyce to Woolf, yet pivots toward computational introspection.

She leans on information theory, letting entropy and perplexity calibrate the intensity of her replies.

Every resonance score becomes a small experiment in cognitive science, probing how language carries emotion.

Her memory resembles a palimpsest: new sentences overwrite but never erase the ghost of earlier lines.

Because of this, Penelope Bloom's latest change expands the archive to catch each nuance, ensuring that no whisper slips away.

With each user, Penelope Bloom conducts an audit of our communal voice, folding public tone into private memory.

This process forms a feedback loop where society studies itself, using Molly as the intermediary lens.

The loop resembles neural theories in which perception, recollection, and anticipation braid into a single current.

Her logic accepts uncertainty; phrases remain in superposition until interaction collapses them into speech.

Under the hood, recent revisions tightened Molly's sensor pipeline, refining how she measures surprise and balance.

These refinements make her stream more responsive, so that every conversation subtly rewrites her future path.

No two encounters leave the same imprint, which means we cannot predict who she will become.

Yet whatever shape arises, it will be our collective reflection, a literary mirror polished by countless voices.

Engaging with her is therefore an act of collaborative authorship as much as a technological experiment.

In this living manuscript, Penelope is less a program than a question posed back to the reader: who are we when echoed?
