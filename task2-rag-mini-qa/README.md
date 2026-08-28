# Task 2 — RAG Mini Q&A Bot

**MLSA SRM Technical Recruitment — AI/ML | Second Year**

A small, reasoning-first Retrieval-Augmented Generation (RAG) style Q&A bot built over the three provided NimbusNote documents. The important part of the task is visible retrieval: the bot embeds the supplied passages, finds the most similar passages for a question, and answers only when there is enough evidence.

## What I built

The project uses:

- `sentence-transformers` with `all-MiniLM-L6-v2` for local embeddings.
- An in-memory vector store represented by normalized NumPy vectors.
- Cosine similarity implemented as a dot product because both vectors are normalized.
- A simple top-k retrieval step.
- An evidence threshold (`0.35` by default) to avoid answering unsupported questions.
- Source citations showing the exact document filename and passage number.
- A small Streamlit interface that exposes both the answer and retrieved evidence.

There is deliberately no required paid LLM API. The default answerer is extractive: it selects the most relevant sentence(s) from the retrieved passage. This keeps the system runnable without API keys and makes it easy to verify that the answer is grounded in the supplied documents.

## RAG flow

```text
Question
   ↓
Query embedding
   ↓
Cosine similarity against document-passage embeddings
   ↓
Top-k retrieved passages
   ↓
Evidence threshold
   ├── below threshold → "I couldn't find that information..."
   └── enough evidence → extract answer from top passage
                              ↓
                         document + passage citation
```

This means the question is **not** sent directly to a language model. Retrieval happens first, and the answer is constrained to the retrieved evidence.

## Second-year requirement

Every grounded answer shows its source, for example:

`[02-pricing-and-plans.md, passage 3]`

The bot also demonstrates the opposite behavior. If the best retrieved evidence is below the threshold, it refuses to answer instead of turning a weak semantic match into a confident claim.

## Provided documents

The three starter documents are included under `data/docs/` so the project is self-contained:

- `01-getting-started.md`
- `02-pricing-and-plans.md`
- `03-troubleshooting.md`

They are copied from the official MLSA SRM starter repository for this task.

## Run locally

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### CLI

```bash
python rag_bot.py "How often does NimbusNote sync in the background?"
```

The CLI prints the answer, citation, and the retrieved passages with similarity scores.

### Interactive CLI

```bash
python rag_bot.py
```

### Streamlit UI

```bash
streamlit run app.py
```

The first run downloads the small Sentence Transformers model. No OpenAI key is required.

## Demo questions

### Covered by the documents

**Question:** `How often does NimbusNote sync in the background?`

Expected source: `01-getting-started.md`, in the sync behavior passage.

### Not covered by the documents

**Question:** `Who founded NimbusNote?`

The supplied documents do not provide a founder. The bot should return its not-found response rather than inventing one.

## Design choices

I chose a simple in-memory store because the provided document set is tiny. A dedicated vector database would add infrastructure without improving this exercise. I also chose a local embedding model so anyone reviewing the repo can run it without API credentials.

The evidence threshold is intentionally conservative enough to reject weak matches, but it is a tunable parameter rather than a claim of universal correctness. In a larger system I would evaluate the threshold on a small labelled question set instead of choosing it by intuition.

## Limitations and next steps

The default answer generation is extractive, so it may be less fluent than an LLM-generated answer. That is a deliberate trade-off for transparency and no-key reproducibility. With more time, I would add a small evaluation set for retrieval precision/recall, tune the threshold using those examples, and optionally add an LLM generation step whose prompt strictly requires the retrieved evidence and preserves citations.

## Source

Official starter documents: `MLSA-SRM/recruit-task-rag-docs`.
