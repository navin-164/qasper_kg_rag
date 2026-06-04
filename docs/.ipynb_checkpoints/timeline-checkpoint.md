# QASPER KG + RAG Timeline

## Phase 1: Setup
- Create Anaconda environment
- Install packages
- Start Neo4j Desktop database
- Create folder structure

## Phase 2: Data Preparation
- Load QASPER dataset
- Extract papers, sections, paragraphs
- Save evaluation questions separately

## Phase 3: Knowledge Graph
- Extract entities from paragraphs
- Extract triples from scientific text
- Ingest papers, sections, paragraphs, entities, triples into Neo4j

## Phase 4: Vector Retrieval
- Build paragraph embeddings
- Create FAISS index
- Test semantic retrieval

## Phase 5: Hybrid RAG
- Combine Neo4j graph retrieval and vector retrieval
- Build answer synthesis
- Expose FastAPI endpoint

## Phase 6: Evaluation
- Run on QASPER QA annotations
- Compute Exact Match and F1
- Save predictions

## Phase 7: Report
- Draw architecture
- Show results
- Discuss limitations and future work