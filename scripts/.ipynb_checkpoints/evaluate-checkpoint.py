from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.rag_server import load_artifacts, answer_question  # noqa: E402


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_answer(s: str) -> str:
    import re
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def f1_score(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = {}
    for tok in pred_tokens:
        common[tok] = common.get(tok, 0) + 1

    num_same = 0
    for tok in gold_tokens:
        if common.get(tok, 0) > 0:
            num_same += 1
            common[tok] -= 1

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def best_metric(pred: str, gold_answers: List[str]) -> Dict[str, float]:
    if not gold_answers:
        gold_answers = ["UNANSWERABLE"]

    em = max(exact_match(pred, g) for g in gold_answers)
    f1 = max(f1_score(pred, g) for g in gold_answers)
    return {"em": em, "f1": f1}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qas", default="data/raw/qasper_eval_qas.jsonl")
    parser.add_argument("--max_questions", type=int, default=None)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--output", default="data/processed/predictions.jsonl")
    args = parser.parse_args()

    qas_path = Path(args.qas)
    qas = read_jsonl(qas_path)
    if args.max_questions is not None:
        qas = qas[: args.max_questions]

    engine = load_artifacts()

    results = []
    em_total = 0.0
    f1_total = 0.0

    for row in tqdm(qas, desc="Evaluating"):
        result = answer_question(
            engine,
            question=row["question"],
            paper_id=row["paper_id"],
            top_k=args.top_k,
        )

        pred = result["answer"]
        gold_answers = row.get("answers", []) or ["UNANSWERABLE"]
        scores = best_metric(pred, gold_answers)

        em_total += scores["em"]
        f1_total += scores["f1"]

        results.append(
            {
                "paper_id": row["paper_id"],
                "question_id": row["question_id"],
                "question": row["question"],
                "prediction": pred,
                "gold_answers": gold_answers,
                "em": scores["em"],
                "f1": scores["f1"],
            }
        )

    n = max(len(qas), 1)
    avg_em = em_total / n
    avg_f1 = f1_total / n

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_path, results)

    print(f"Exact Match: {avg_em:.4f}")
    print(f"F1:          {avg_f1:.4f}")
    print(f"Saved predictions to {out_path}")


if __name__ == "__main__":
    main()