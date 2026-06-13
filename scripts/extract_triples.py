from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

try:
    import spacy
except Exception:
    spacy = None

# --- NEW LLM IMPORTS ---
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
# -----------------------


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


PATTERNS: List[Tuple[str, str]] = [
    (r"(.+?)\s+is used for\s+(.+)", "USED_FOR"),
    (r"(.+?)\s+was used for\s+(.+)", "USED_FOR"),
    (r"(.+?)\s+is trained on\s+(.+)", "TRAINED_ON"),
    (r"(.+?)\s+was trained on\s+(.+)", "TRAINED_ON"),
    (r"(.+?)\s+is based on\s+(.+)", "BASED_ON"),
    (r"(.+?)\s+was based on\s+(.+)", "BASED_ON"),
    (r"(.+?)\s+outperforms\s+(.+)", "OUTPERFORMS"),
    (r"(.+?)\s+achieves\s+(.+)", "ACHIEVES"),
    (r"(.+?)\s+improves\s+(.+)", "IMPROVES"),
    (r"(.+?)\s+consists of\s+(.+)", "CONSISTS_OF"),
    (r"(.+?)\s+includes\s+(.+)", "INCLUDES"),
    (r"(.+?)\s+compares with\s+(.+)", "COMPARES_WITH"),
]


def load_nlp():
    if spacy is None:
        return None
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        return None


def normalize_text(text: str) -> str:
    return " ".join((text or "").split()).strip()


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


def split_sentences(text: str, nlp=None) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    if nlp is not None:
        doc = nlp(text)
        return [normalize_text(sent.text) for sent in doc.sents if normalize_text(sent.text)]
    return [normalize_text(s) for s in re.split(r"(?<=[.!?])\s+", text) if normalize_text(s)]


def clean_phrase(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"^[\(\[\{]+|[\)\]\},.;:]+$", "", text)
    text = re.sub(r"^(the|a|an)\s+", "", text, flags=re.I)
    return normalize_text(text)


def extract_entities(text: str, nlp=None) -> List[str]:
    text = normalize_text(text)
    ents = set()

    if nlp is not None:
        doc = nlp(text)
        for ent in doc.ents:
            val = clean_phrase(ent.text)
            if len(val) >= 2:
                ents.add(val)

    if not ents:
        for m in re.finditer(r"\b[A-Z][A-Za-z0-9\-]+(?:\s+[A-Z][A-Za-z0-9\-]+)*\b", text):
            val = clean_phrase(m.group(0))
            if len(val) >= 2:
                ents.add(val)

    return sorted(ents)


def pattern_triples(sentence: str) -> List[Dict[str, Any]]:
    triples = []
    for pattern, relation in PATTERNS:
        m = re.search(pattern, sentence, flags=re.I)
        if m:
            s = clean_phrase(m.group(1))
            o = clean_phrase(m.group(2))
            if s and o and s.lower() != o.lower():
                triples.append(
                    {
                        "subject": s,
                        "relation": relation,
                        "object": o,
                        "confidence": 0.75,
                    }
                )
    return triples


def cooccurrence_triples(sentence: str, nlp=None) -> List[Dict[str, Any]]:
    ents = extract_entities(sentence, nlp=nlp)
    triples = []
    if len(ents) < 2:
        return triples

    for i in range(len(ents)):
        for j in range(i + 1, len(ents)):
            if ents[i].lower() != ents[j].lower():
                triples.append(
                    {
                        "subject": ents[i],
                        "relation": "CO_OCCURS_WITH",
                        "object": ents[j],
                        "confidence": 0.3,
                    }
                )
    return triples


# --- NEW LLM SETUP FUNCTION ---
def setup_llm_extractor(model_name="llama3"):
    """Initializes the local Ollama LLM and enforces JSON schema output."""
    llm = ChatOllama(model=model_name, temperature=0.0, format="json")
    parser = JsonOutputParser()
    
    prompt = PromptTemplate(
        template="""Extract the core semantic relationships from the following academic sentence.
        Format the output strictly as a JSON array of objects. Each object MUST contain exactly these keys: 'subject', 'relation', 'object'.
        
        Rules:
        1. Subjects and Objects should be concise noun phrases.
        2. Relations should be clear verbs, uppercase, with underscores for spaces (e.g., 'EVALUATES_ON').
        3. Do not invent information.
        
        Sentence: {text}
        """,
        input_variables=["text"]
    )
    return prompt | llm | parser
# ------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/paragraphs.jsonl")
    parser.add_argument("--output", default="data/triples/triples.jsonl")
    parser.add_argument("--max_rows", type=int, default=None)
    # Added argument to specify the LLM model
    parser.add_argument("--model", type=str, default="llama3", help="Ollama model to use") 
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    nlp = load_nlp()
    rows = read_jsonl(input_path)
    if args.max_rows is not None:
        rows = rows[: args.max_rows]

    out = []
    
    # Initialize the LLM Chain
    print(f"Initializing local LLM extractor using model: {args.model}")
    llm_extractor = setup_llm_extractor(model_name=args.model)

    for row in tqdm(rows, desc="Extracting triples"):
        sentences = split_sentences(row["text"], nlp=nlp)

        for sent in sentences:
            
            # --- MODIFIED EXTRACTION LOGIC ---
            try:
                # 1. Invoke the LLM
                llm_response = llm_extractor.invoke({"text": sent})
                print(f"\n[RAW LLM OUTPUT for '{sent[:30]}...']: {llm_response}")
                
                # 2. Smart Unwrapping (Handles "data" wrappers)
                triples = []
                if isinstance(llm_response, list):
                    triples = llm_response
                elif isinstance(llm_response, dict):
                    # Check if it's a single triple directly
                    if "subject" in [k.lower() for k in llm_response.keys()]:
                        triples = [llm_response]
                    else:
                        # Dig into keys like 'data' or 'semantic_relationships' to find the list
                        for val in llm_response.values():
                            if isinstance(val, list):
                                triples.extend(val)
                
                # 3. Robust key mapping
                for t in triples:
                    if not isinstance(t, dict): # Skip weirdly formatted items
                        continue
                        
                    safe_t = {str(k).lower(): v for k, v in t.items()}
                    
                    subj = safe_t.get("subject", safe_t.get("entity1", safe_t.get("source")))
                    rel = safe_t.get("relation", safe_t.get("predicate", safe_t.get("action")))
                    obj = safe_t.get("object", safe_t.get("entity2", safe_t.get("target")))
                    
                    if subj and rel and obj:
                        out.append({
                            "paper_id": row["paper_id"],
                            "paragraph_id": row["paragraph_id"],
                            "section_idx": row["section_idx"],
                            "section_name": row["section_name"],
                            "sentence": sent,
                            "subject": clean_phrase(str(subj)),
                            "relation": str(rel).strip().upper().replace(" ", "_"),
                            "object": clean_phrase(str(obj)),
                            "confidence": 0.95
                        })
            except Exception as e:
                print(f"\n[LLM ERROR]: {e}")
                triples = pattern_triples(sent)
            # ---------------------------------
            # ---------------------------------

            for t in triples:
                # Retains original dictionary structure
                if "subject" in t and "relation" in t and "object" in t:
                    out.append(
                        {
                            "paper_id": row["paper_id"],
                            "paragraph_id": row["paragraph_id"],
                            "section_idx": row["section_idx"],
                            "section_name": row["section_name"],
                            "sentence": sent,
                            "subject": t["subject"],
                            "relation": t["relation"],
                            "object": t["object"],
                            "confidence": t.get("confidence", 0.75),
                        }
                    )

    write_jsonl(output_path, out)
    print(f"Saved {len(out)} triples to {output_path}")


if __name__ == "__main__":
    main()