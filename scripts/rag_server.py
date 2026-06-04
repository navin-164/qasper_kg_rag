from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from neo4j import GraphDatabase
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


STOPWORDS = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "by",
    "is", "are", "was", "were", "be", "been", "this", "that", "these", "those",
    "what", "which", "who", "whom", "when", "where", "why", "how", "does", "do",
    "did", "can", "could", "should", "would", "may", "might", "will"
}


class QuestionRequest(BaseModel):
    question: str
    paper_id: Optional[str] = None
    top_k: int = 5


class AnswerResponse(BaseModel):
    question: str
    answer: str
    evidence_paragraphs: List[Dict[str, Any]]
    graph_facts: List[Dict[str, Any]]
    context: str


@dataclass
class RagArtifacts:
    driver: Any
    database: str
    embedder: SentenceTransformer
    faiss_index: faiss.Index
    paragraphs: List[Dict[str, Any]]


def normalize_text(text: str) -> str:
    return " ".join((text or "").split()).strip()


def tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9]+", (text or "").lower())
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def extract_question_entities(question: str) -> List[str]:
    question = normalize_text(question)
    ents = []

    for m in re.finditer(r"\b[A-Z][A-Za-z0-9\-]+(?:\s+[A-Z][A-Za-z0-9\-]+)*\b", question):
        val = normalize_text(m.group(0))
        if len(val) > 1:
            ents.append(val)

    if not ents:
        ents = [w for w in re.findall(r"[A-Za-z0-9\-]+", question) if len(w) > 2]

    return sorted(set(ents))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_artifacts() -> RagArtifacts:
    load_dotenv()

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password123")
    database = os.getenv("NEO4J_DATABASE", "neo4j")
    embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    embedder = SentenceTransformer(embedding_model)

    faiss_path = PROJECT_ROOT / "data/faiss/paragraph.index"
    meta_path = PROJECT_ROOT / "data/embeddings/paragraph_meta.jsonl"

    if not faiss_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {faiss_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Paragraph metadata not found: {meta_path}")

    index = faiss.read_index(str(faiss_path))
    paragraphs = read_jsonl(meta_path)

    return RagArtifacts(
        driver=driver,
        database=database,
        embedder=embedder,
        faiss_index=index,
        paragraphs=paragraphs,
    )


def search_vector(artifacts: RagArtifacts, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
    qvec = artifacts.embedder.encode([question], show_progress_bar=False)
    qvec = np.asarray(qvec, dtype="float32")
    faiss.normalize_L2(qvec)

    scores, idxs = artifacts.faiss_index.search(qvec, top_k)
    results = []

    for score, idx in zip(scores[0], idxs[0]):
        if idx < 0 or idx >= len(artifacts.paragraphs):
            continue
        row = dict(artifacts.paragraphs[idx])
        row["score"] = float(score)
        results.append(row)

    return results


def search_graph(artifacts: RagArtifacts, question: str, paper_id: Optional[str], top_k: int = 5) -> List[Dict[str, Any]]:
    ents = extract_question_entities(question)
    if not ents:
        return []

    query = """
    MATCH (p:Paragraph)-[:MENTIONS]->(e:Entity)
    WHERE toLower(e.name) IN $entities
      AND ($paper_id IS NULL OR p.paper_id = $paper_id)
    RETURN DISTINCT p.paragraph_id AS paragraph_id,
                    p.paper_id AS paper_id,
                    p.section_idx AS section_idx,
                    p.section_id AS section_id,
                    p.paragraph_idx AS paragraph_idx,
                    p.text AS text
    LIMIT $limit
    """

    rows = []
    with artifacts.driver.session(database=artifacts.database) as session:
        result = session.run(
            query,
            entities=[e.lower() for e in ents],
            paper_id=paper_id,
            limit=top_k,
        )
        for rec in result:
            rows.append(
                {
                    "paragraph_id": rec["paragraph_id"],
                    "paper_id": rec["paper_id"],
                    "section_idx": rec["section_idx"],
                    "section_id": rec["section_id"],
                    "paragraph_idx": rec["paragraph_idx"],
                    "text": rec["text"],
                    "source": "graph",
                }
            )

    return rows


def search_graph_facts(artifacts: RagArtifacts, question: str, paper_id: Optional[str], top_k: int = 10) -> List[Dict[str, Any]]:
    ents = extract_question_entities(question)
    if not ents:
        return []

    query = """
    MATCH (a:Entity)-[r:RELATED_TO]->(b:Entity)
    WHERE (toLower(a.name) IN $entities OR toLower(b.name) IN $entities)
      AND ($paper_id IS NULL OR r.paper_id = $paper_id)
    RETURN a.name AS subject,
           r.relation AS relation,
           b.name AS object,
           r.confidence AS confidence,
           r.sentence AS sentence
    LIMIT $limit
    """

    facts = []
    with artifacts.driver.session(database=artifacts.database) as session:
        result = session.run(
            query,
            entities=[e.lower() for e in ents],
            paper_id=paper_id,
            limit=top_k,
        )
        for rec in result:
            facts.append(
                {
                    "subject": rec["subject"],
                    "relation": rec["relation"],
                    "object": rec["object"],
                    "confidence": rec["confidence"],
                    "sentence": rec["sentence"],
                }
            )
    return facts


def deduplicate_paragraphs(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for item in items:
        pid = item.get("paragraph_id")
        if pid in seen:
            continue
        seen.add(pid)
        out.append(item)
    return out


def rank_sentences(question: str, context: str, max_sentences: int = 3) -> str:
    q_tokens = set(tokenize(question))
    if not q_tokens:
        return context[:1000]

    sentences = re.split(r"(?<=[.!?])\s+", context)
    scored = []

    for sent in sentences:
        s = normalize_text(sent)
        if not s:
            continue
        s_tokens = set(tokenize(s))
        score = len(q_tokens & s_tokens)
        if score > 0:
            scored.append((score, s))

    if not scored:
        return sentences[0] if sentences and sentences[0] else context[:1000]

    scored.sort(key=lambda x: (-x[0], len(x[1])))
    chosen = [s for _, s in scored[:max_sentences]]
    return " ".join(chosen)


def build_context(vector_hits: List[Dict[str, Any]], graph_hits: List[Dict[str, Any]], facts: List[Dict[str, Any]]) -> str:
    parts = []

    if facts:
        parts.append("Graph facts:")
        for f in facts:
            parts.append(f"- {f['subject']} [{f['relation']}] {f['object']}")

    if graph_hits:
        parts.append("\nGraph paragraphs:")
        for i, row in enumerate(graph_hits, 1):
            parts.append(
                f"[G{i}] paper={row.get('paper_id')} section={row.get('section_idx')} para={row.get('paragraph_idx')}: {row.get('text')}"
            )

    if vector_hits:
        parts.append("\nVector paragraphs:")
        for i, row in enumerate(vector_hits, 1):
            parts.append(
                f"[V{i}] paper={row.get('paper_id')} section={row.get('section_idx')} para={row.get('paragraph_idx')}: {row.get('text')}"
            )

    return "\n".join(parts).strip()


def answer_question(artifacts: RagArtifacts, question: str, paper_id: Optional[str] = None, top_k: int = 5) -> Dict[str, Any]:
    vector_hits = search_vector(artifacts, question, top_k=top_k)
    graph_hits = search_graph(artifacts, question, paper_id=paper_id, top_k=top_k)
    facts = search_graph_facts(artifacts, question, paper_id=paper_id, top_k=top_k)

    combined = deduplicate_paragraphs(graph_hits + vector_hits)
    context = build_context(vector_hits=vector_hits, graph_hits=graph_hits, facts=facts)

    if combined:
        merged_text = " ".join([r["text"] for r in combined[: top_k * 2]])
        answer = rank_sentences(question, merged_text, max_sentences=3)
    else:
        answer = "I could not find enough supporting text in the paper."

    return {
        "question": question,
        "answer": answer,
        "evidence_paragraphs": combined,
        "graph_facts": facts,
        "context": context,
    }


app = FastAPI(title="QASPER KG + RAG")


ENGINE: Optional[RagArtifacts] = None


@app.on_event("startup")
def startup_event():
    global ENGINE
    ENGINE = load_artifacts()


@app.on_event("shutdown")
def shutdown_event():
    global ENGINE
    if ENGINE and ENGINE.driver:
        ENGINE.driver.close()
        ENGINE = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AnswerResponse)
def ask(req: QuestionRequest):
    if ENGINE is None:
        raise HTTPException(status_code=500, detail="RAG engine not initialized")

    result = answer_question(
        ENGINE,
        question=req.question,
        paper_id=req.paper_id,
        top_k=req.top_k,
    )
    return result