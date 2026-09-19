# SQuAD RAG-System

Et **simpelt RAG-system (Retrieval-Augmented Generation)** til at svare på spørgsmål fra [SQuAD-dataset](https://rajpurkar.github.io/SQuAD-explorer/) ved hjælp af **Chroma DB** (vektordatabase) og **Ollama** (lokal LLM-server).
Projektet er et **læringsprojekt inden for AI** og demonstrerer, hvordan man kombinerer søgning i struktureret tekst med sprogmodeller.

- **Retrieval**: Søger i SQuAD-dataset ved hjælp af embeddings (Sentence Transformers).
- **Generation**: Generer svar med Qwen2.5-3B (eller andre Ollama-modeller).
- **Evaluering**: Tester systemet med 10 spørgsmål og rapporterer præcision.
- **Interaktivt mode**: Stil spørgsmål i realtid.

- **OS**: Linux, macOS, eller Windows (med WSL).
- **Hardware**: Testet på **Raspberry Pi 5 (8GB RAM)** med Linux.
- **Python**: 3.8+.
