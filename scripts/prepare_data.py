from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from datasets import load_dataset
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATASET_NAME = "allenai/qasper"


def normalize_text(text: str) -> str:
    return " ".join((text or "").split()).strip()


def ensure_dirs(base: Path) -> None:
    for sub in [
        "data/raw",
        "data/processed",
        "data/triples",
        "data/embeddings",
        "data/faiss",
    ]:
        (base / sub).mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_sections_and_paragraphs(paper: Dict[str, Any]) -> List[Dict[str, Any]]:
    full_text = paper.get("full_text", {})
    section_names = full_text.get("section_name", [])
    section_paragraphs = full_text.get("paragraphs", [])

    records = []
    for sec_idx, paragraphs in enumerate(section_paragraphs):
        section_name = section_names[sec_idx] if sec_idx < len(section_names) else f"section_{sec_idx}"

        for para_idx, para in enumerate(paragraphs):
            text = normalize_text(para)
            if not text:
                continue

            records.append(
                {
                    "paper_id": str(paper["id"]),
                    "paper_title": normalize_text(paper.get("title", "")),
                    "abstract": normalize_text(paper.get("abstract", "")),
                    "section_idx": sec_idx,
                    "section_name": section_name,
                    "paragraph_idx": para_idx,
                    "paragraph_id": f"{paper['id']}::{sec_idx}::{para_idx}",
                    "text": text,
                }
            )
    return records


def extract_questions_for_eval(paper: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Save QA annotations only for evaluation.
    Do NOT ingest these into Neo4j.
    """
    qas = paper.get("qas", {})
    questions = qas.get("question", [])
    question_ids = qas.get("question_id", [])
    answers_groups = qas.get("answers", [])

    rows = []
    for q_idx, q_text in enumerate(questions):
        answer_group = answers_groups[q_idx] if q_idx < len(answers_groups) else {}
        answer_items = answer_group.get("answer", []) or []

        answer_texts = []
        evidence_texts = []

        for ans in answer_items:
            if ans.get("unanswerable", False):
                answer_texts.append("UNANSWERABLE")
            elif ans.get("free_form_answer"):
                answer_texts.append(normalize_text(ans["free_form_answer"]))
            elif ans.get("extractive_spans"):
                spans = [normalize_text(s) for s in ans.get("extractive_spans", []) if normalize_text(s)]
                answer_texts.extend(spans)
            elif ans.get("yes_no") is True:
                answer_texts.append("Yes")
            elif ans.get("yes_no") is False:
                answer_texts.append("No")

            for ev in ans.get("evidence", []) or []:
                ev_text = normalize_text(ev)
                if ev_text:
                    evidence_texts.append(ev_text)

        rows.append(
            {
                "paper_id": str(paper["id"]),
                "question_id": str(question_ids[q_idx]) if q_idx < len(question_ids) else f"{paper['id']}_q{q_idx}",
                "question": normalize_text(q_text),
                "answers": sorted(set(a for a in answer_texts if a)),
                "evidence": sorted(set(e for e in evidence_texts if e)),
            }
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    parser.add_argument("--max_papers", type=int, default=None)
    parser.add_argument("--output_dir", default=".")
    args = parser.parse_args()

    base = Path(args.output_dir).resolve()
    ensure_dirs(base)

    ds = load_dataset(DATASET_NAME, split=args.split)
    if args.max_papers is not None:
        ds = ds.select(range(min(args.max_papers, len(ds))))

    paper_rows = []
    paragraph_rows = []
    eval_rows = []

    for paper in tqdm(ds, desc=f"Preparing QASPER/{args.split}"):
        paper_rows.append(
            {
                "paper_id": str(paper["id"]),
                "title": normalize_text(paper.get("title", "")),
                "abstract": normalize_text(paper.get("abstract", "")),
                "source_split": args.split,
            }
        )

        paragraph_rows.extend(extract_sections_and_paragraphs(paper))
        eval_rows.extend(extract_questions_for_eval(paper))

    write_jsonl(base / "data/raw/papers.jsonl", paper_rows)
    write_jsonl(base / "data/processed/paragraphs.jsonl", paragraph_rows)
    write_jsonl(base / "data/raw/qasper_eval_qas.jsonl", eval_rows)

    print(f"Saved papers: {len(paper_rows)}")
    print(f"Saved paragraphs: {len(paragraph_rows)}")
    print(f"Saved eval QA rows: {len(eval_rows)}")


if __name__ == "__main__":
    main()