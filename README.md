# Hypothetica

**AI-Powered Research Originality Assessment with Multi-Agent RAG**

## Overview

Hypothetica is an AI-powered system that evaluates the originality of research ideas by comparing them against existing academic literature. Instead of simple text similarity or plagiarism detection, Hypothetica performs conceptual, sentence-level novelty analysis using a multi-agent architecture, Retrieval-Augmented Generation (RAG), and evidence-grounded reasoning.

The system is designed to help researchers, students, and early-stage innovators answer a critical question before committing months of work:

> "Is my research idea actually novel?"

## What Makes Hypothetica Different?

**Most existing tools:**
- Detect text overlap, not idea overlap
- Provide opaque similarity scores
- Do not explain why something is considered unoriginal

**Hypothetica instead:**
- Evaluates conceptual originality, not plagiarism
- Produces sentence-level explanations
- Links every originality claim to retrieved academic evidence
- Uses multi-agent reasoning instead of a single LLM prompt

## Core Features

### Conceptual Originality Assessment
- Compares a user's research idea against recent academic papers from arXiv
- Identifies overlaps in:
  - Technical problem definition
  - Methodology
  - Application domain
  - Innovation claims

### Multi-Agent Architecture

Specialized LLM agents handle different stages:
- **Keyword Agent** – generates structured search queries
- **Relevance Ranking Agent** – selects the most relevant papers
- **Layer-1 Analysis Agent** – evaluates each paper independently
- **Layer-2 Aggregation Agent** – synthesizes global originality insights

Each agent operates with task-specific parameters for reliability and determinism.

### Structured, Explainable Scoring
- Outputs a 0–100 originality score
- Uses Likert-scale categorical judgments instead of unreliable continuous scoring
- Assigns sentence-level originality labels:
  - **High originality**
  - **Medium originality**
  - **Low originality**

Each flagged sentence is linked to specific evidence from retrieved papers.

### Retrieval-Augmented Generation (RAG)
- Papers are embedded using E5 embeddings
- Stored and retrieved via ChromaDB
- Ensures all LLM reasoning is grounded in real academic content

## System Architecture (High-Level)

```
User Research Idea
        ↓
Keyword Generation Agent
        ↓
arXiv Paper Retrieval
        ↓
Relevance Ranking Agent
        ↓
Layer-1 Per-Paper Analysis (5 papers)
        ↓
Layer-2 Global Aggregation
        ↓
Originality Score + Sentence-Level Report
```

## Technology Stack

### Core
- Python 3.10
- Google Gemini 2.5 Flash
- Retrieval-Augmented Generation (RAG)

### Retrieval & Embeddings
- arXiv API
- ChromaDB
- E5-base-v2 embeddings

### Document Processing
- Docling (PDF → structured Markdown)
- Custom heading extraction & section filtering

### Architecture
- Modular, agent-based design
- Structured JSON outputs enforced at every stage

### Frontend
- React single-page web interface
- Color-coded sentence-level feedback
- Expandable evidence views

## Evaluation Highlights
- Tested on 10 diverse research ideas
- Correctly assigns very low scores to:
  - Image classification
  - Sentiment analysis
  - YOLO reproductions
- Rewards true novelty and novel domain combinations
- Average cost per evaluation: ~$0.013
- Stable, deterministic scoring across repeated runs

## Known Limitations
- Current corpus is limited to academic literature (arXiv)
- Does not detect commercial product cloning
  - e.g., a "Facebook-like social network" may score high
- Future versions will integrate:
  - IEEE / ACM corpora
  - Product & patent databases

## Future Work
- Multi-corpus retrieval (ACM, IEEE, OpenAlex)
- Domain-specific originality agents
- Longitudinal novelty tracking
- Cost-optimized local model support
- Patent & commercial product awareness

## Contributors
- **Ahmet Alp Malkoç** — Multi-agent pipeline, originality scoring, RAG architecture
- **Harun Hüdai Tan** — System design, evaluation framework
- **Kutay Becerir** — Frontend & UX
- **Baran Erol** — Research support & system testing
- **Murat Karakaya** — Academic supervision

## Academic Context
- **Course:** SENG 491 – Senior Project I
- **Institution:** TED University
- **Supervisor:** Dr. Murat Karakaya

## License

This project is currently intended for research and academic use. Licensing details can be added once the project is open-sourced publicly.
