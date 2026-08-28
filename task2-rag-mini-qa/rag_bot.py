"""NimbusNote Mini RAG Q&A Bot.

A small, explainable RAG implementation:
1. load the provided documents
2. split them into passages
3. embed passages with sentence-transformers
4. retrieve by cosine similarity
5. answer only from retrieved evidence
6. cite the source document and passage

The default answerer is extractive, so the project works without an API key.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_THRESHOLD = 0.35
DEFAULT_TOP_K = 3


@dataclass(frozen=True)
class Passage:
    doc_name: str
    passage_id: int
    text: str


class MiniRAG:
    def __init__(
        self,
        docs_dir: str | Path = "data/docs",
        model_name: str = DEFAULT_MODEL,
        threshold: float = DEFAULT_THRESHOLD,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        self.docs_dir = Path(docs_dir)
        self.threshold = threshold
        self.top_k = top_k
        self.model = SentenceTransformer(model_name)
        self.passages = self._load_passages()
        self.embeddings = self.model.encode(
            [p.text for p in self.passages],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def _load_passages(self) -> list[Passage]:
        passages: list[Passage] = []
        for path in sorted(self.docs_dir.glob("*.md")):
            text = path.read_text(encoding="utf-8")
            # Markdown headings are retained with their following paragraph/list
            # content so each retrieved unit has enough context to answer from.
            blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
            for index, block in enumerate(blocks, start=1):
                if block.startswith("# "):
                    continue
                passages.append(Passage(path.name, index, block))
        if not passages:
            raise FileNotFoundError(f"No Markdown documents found in {self.docs_dir}")
        return passages

    def retrieve(self, question: str) -> list[tuple[Passage, float]]:
        query_embedding = self.model.encode(
            [question], normalize_embeddings=True, show_progress_bar=False
        )[0]
        scores = np.asarray(self.embeddings) @ query_embedding
        order = np.argsort(scores)[::-1][: self.top_k]
        return [(self.passages[i], float(scores[i])) for i in order]

    def answer(self, question: str) -> dict:
        retrieved = self.retrieve(question)
        if not retrieved or retrieved[0][1] < self.threshold:
            return {
                "answer": "I couldn't find that information in the provided documents.",
                "grounded": False,
                "sources": retrieved,
            }

        # Extractive answer: choose the most relevant sentence(s) from the top
        # evidence rather than inventing a response outside the documents.
        top_passage, top_score = retrieved[0]
        sentences = self._sentences(top_passage.text)
        question_terms = self._terms(question)
        ranked = sorted(
            sentences,
            key=lambda sentence: self._sentence_score(sentence, question_terms),
            reverse=True,
        )
        selected = [s for s in ranked[:2] if self._sentence_score(s, question_terms) > 0]
        answer_text = " ".join(selected) if selected else top_passage.text
        return {
            "answer": answer_text,
            "grounded": True,
            "sources": retrieved,
            "citation": f"[{top_passage.doc_name}, passage {top_passage.passage_id}]",
            "top_score": top_score,
        }

    @staticmethod
    def _terms(text: str) -> set[str]:
        stop = {
            "what", "which", "when", "where", "how", "does", "do", "is", "are",
            "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with",
            "can", "i", "my", "it", "this", "that", "be", "from", "as", "if",
        }
        return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in stop}

    @classmethod
    def _sentence_score(cls, sentence: str, question_terms: set[str]) -> int:
        return len(cls._terms(sentence) & question_terms)

    @staticmethod
    def _sentences(text: str) -> list[str]:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.replace("\n", " ")) if s.strip()]


def print_result(bot: MiniRAG, question: str) -> None:
    result = bot.answer(question)
    print("\nQUESTION:", question)
    print("\nANSWER:", result["answer"])
    if result["grounded"]:
        print("CITATION:", result["citation"])
    else:
        print("CITATION: None — evidence threshold not met.")

    print("\nRETRIEVED PASSAGES:")
    for passage, score in result["sources"]:
        print(f"- {passage.doc_name} | passage {passage.passage_id} | cosine={score:.3f}")
        print(f"  {passage.text[:220].replace(chr(10), ' ')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="NimbusNote Mini RAG Q&A Bot")
    parser.add_argument("question", nargs="*", help="question to ask")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()

    bot = MiniRAG(threshold=args.threshold, top_k=args.top_k)
    question = " ".join(args.question).strip()
    if question:
        print_result(bot, question)
        return

    print("NimbusNote Mini RAG Q&A Bot")
    print("Type a question, or Ctrl+C to exit.")
    while True:
        try:
            question = input("\n> ").strip()
            if question:
                print_result(bot, question)
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break


if __name__ == "__main__":
    main()
