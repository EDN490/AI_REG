#!/usr/bin/env python3
"""
SQuAD RAG-System
================
Sidst opdateret: 2026-09-18 20:04:00

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
HVORDAN RAG FUNGERER
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
RAG (Retrieval-Augmented Generation) er en arkitektur, der kombinerer:
1. **Retrieval**: Søgning i en vektordatabase (Chroma DB) for at finde relevant kontekst.
2. **Augmented Generation**: LLM'en (Qwen2.5-3B) generer svar baseret på den fundne kontekst.

Workflow:
   [Spørgsmål] → [Konverter til vektor] → [Søg i Chroma DB] → [Top-N chunks]
        ↓
   [Send chunks til LLM] → [Generer svar baseret på kontekst]

Fordele ved RAG:
- **Reducerer hallucinationer**: LLM'en svare udelukkende baseret på kontekst, ikke sin træning.
- **Aktuelle oplysninger**: Kan svare baseret på specifikke data (f.eks. SQuAD-dataset).
- **Transparens**: Svaret kan spores tilbage til konteksten.

Begrænsninger:
- **Afhængig af kontekstkvalitet**: Hvis konteksten mangler svaret, kan LLM'en ikke svare korrekt.
- **Chunking-problemer**: For små chunks kan splitte vigtige oplysninger.
- **Embeddings-præcision**: Dårlige embeddings giver dårlige matches.

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
SQuAD-DATASET
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
Direkte download (JSON):
   https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/train-v1.1.json

Officiel hjemmeside:
   https://rajpurkar.github.io/SQuAD-explorer/

Hvad er SQuAD?
   SQuAD (Stanford Question Answering Dataset) er et dataset med 100.000+
   spørgsmål-svar-par, baseret på Wikipedia-artikler. Hvert spørgsmål har et
   korrekt svar, der er et direkte uddrag fra en kontekst-tekst. Datasetet
   bruges til at træne og teste modeller i tekstforståelse (reading comprehension).

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
CENTRALE KONCEPTER
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

Embeddings:
   Konverterer tekst til numeriske vektorer (talrækker), så computeren kan
   forstå tekstens betydning og sammenligne tekster. I dette program bruger
   vi modellen 'all-MiniLM-L6-v2' (384 dimensioner, 80MB) til at omdanne SQuAD-tekster.
   Andre populære modeller: 'all-mpnet-base-v2' (768 dimensioner, 420MB).

Chroma DB:
   En vektordatabase, der effektivt gemmer og søger i embeddings. Når du
   stiller et spørgsmål, konverteres det til en vektor, og Chroma DB finder
   hurtigt de mest lignende vektorer ved at bruge cosinus-lighed (cosine similarity).
   Cosinus-lighed måler vinklen mellem vektorer: 1.0 = identisk retning (perfekt match),
   0.0 = ingen lighed, -1.0 = modsat retning.

LLM/Ollama:
   LLM (Large Language Model) er en AI-model, der kan forstå og generere
   menneskelig tekst. Ollama er en lokal server, der gør det nemt at køre
   LLM'er på din egen maskine. I dette program bruger vi Qwen2.5-3B til at
   generere svar baseret på den kontekst, der er fundet via Chroma DB.

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
KOMPONENTER OG DERES VIRKEN
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

Komponent: SQuAD-dataset
Rolle: Leverer spørgsmål + kontekst til RAG-systemet
Teknologi: JSON-fil

Komponent: Chroma DB
Rolle: Gemmer og søger i vektorer (embeddings)
Teknologi: Vektordatabase

Komponent: Ollama
Rolle: Kører LLM-modellen lokalt og generer svar
Teknologi: LLM-server

Komponent: text-splitters
Rolle: Opdeler lange tekster i chunks
Teknologi: langchain-text-splitters

Komponent: Sentence Transformers
Rolle: Konverterer tekst til vektorer
Teknologi: all-MiniLM-L6-v2

Komponent: RAG-logik
Rolle: Kombinerer søgning + generering
Teknologi: Python-script

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
DESIGN-BESKRIVELSE
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

Arkitektur:
  SQuAD-dataset (JSON)
       ↓
  [Load Subset] -> Indlæser 1.000 spørgsmål + kontekster
       ↓
  [Clean & Chunk] -> Opdeler tekster i 500-tegn chunks
       ↓
  [Embeddings] -> Konverterer chunks til vektorer (all-MiniLM-L6-v2)
       ↓
  [Chroma DB] -> Gemmer vektorer i lokal database
       ↓
  [RAG Query] -> Søger i Chroma DB + sender top-3 chunks til LLM
       ↓
  [Ollama (Qwen2.5-3B)] -> Generer svar baseret på kontekst
       ↓
  [Output] -> Returnerer svar til brugeren

Workflow:
  1. Forberedelse: Indlæs SQuAD -> Chunk -> Embed -> Gem i Chroma DB
  2. Spørgsmål: Søg i Chroma DB -> Send kontekst til LLM -> Få svar
  3. Evaluering: Sammenlign LLM-svar med korrekte svar fra SQuAD

Optimeringer:
  - Streaming JSON-parsing (ijson) for at spare RAM
  - Batch-processing af embeddings (32 chunks ad gangen)

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
EKSEMPLER
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
Når systemet køres, kan du stille spørgsmål som:
- "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?"
  → Svar: "Saint Bernadette Soubirous"
- "How many student newspapers are found at Notre Dame?"
  → Svar: "Three"
- "When did the Scholastic Magazine of Notre Dame begin publishing?"
  → Svar: "September 1876"

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
FEJLSØGNING
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
- [ModuleNotFoundError]:
  - Sikre, at alle pakker er installeret: `pip install ollama chromadb sentence-transformers ijson langchain-text-splitters`.
- [Chroma DB returnerer ingen resultater]:
  - Tjek, at `squad_rag_db` eksisterer i den aktuelle mappe.
  - Prøv at slette og genoprette Chroma DB: `rm -rf squad_rag_db` og kør programmet igen.
- [LLM’en siger "Jeg ved det ikke" for ofte]:
  - Øg `n_results` i `ask_rag` (f.eks. fra 3 til 5).
  - Øg `chunk_size` (f.eks. fra 500 til 1000) for at bevare mere kontekst.
- [Forkert svar]:
  - Tjek om konteksten indeholder det korrekte svar. Hvis ja, juster prompten til at være mere specifik.

Output-format under evaluering:
- [OK]: RAG-svaret matchede det korrekte svar (uanset case).
- [ERR]: RAG-svaret var forkert eller ufuldstændigt.
- Præcision: Procentdel af korrekte svar (f.eks. 50.0% = 5/10 korrekte).

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

ANVENDELSE:
- Kundesupport: Automatiser FAQ-svar baseret på produktdokumentation.
- Intern videnbase: Gør medarbejdere i stand til hurtigt at finde HR/IT-oplysninger.
- Faglig støtte: Hjælp eksperter (læger, advokater) med at finde præcise oplysninger i store dokumenter.

KORT INSTALLATIONSVEJLEDNING:
1. Installer Ollama: curl -fsSL https://ollama.ai/install.sh | sh
2. Træk model: ollama pull qwen2.5:3b
3. Opret virtuelt miljø (anbefalet):
   python -m venv .venv
   Aktivér det:
   - Linux/macOS: source .venv/bin/activate
   - Windows: .venv\\Scripts\\activate
4. Installer Python-afhængigheder:
   pip install ollama chromadb sentence-transformers ijson langchain-text-splitters
5. Kør: python3 squad_rag.py

Krav: Linux, macOS, eller Windows (med WSL) + Python 3.8+
Testet på: Raspberry Pi 5 (8GB RAM) med Ubuntu 24.04
"""

import os
import re
import ijson
from typing import List, Tuple

try:
    from ollama import Client
    from sentence_transformers import SentenceTransformer
    import chromadb
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError as e:
    print(f"[ERR] Manglende bibliotek: {e}")
    print("Installer med: pip install ollama chromadb sentence-transformers ijson langchain-text-splitters")
    exit(1)

# --- Konfiguration ---
CONFIG = {
    "squad_file": "train-v1.1.json",          # Filnavn for SQuAD-dataset (JSON)
    "max_questions": 1000,                    # Maksimal antal spørgsmål at indlæse fra SQuAD
    "chunk_size": 500,                        # Størrelse på hver tekst-chunk (tegn)
    "chunk_overlap": 50,                      # Overlap mellem chunks (tegn) for at bevare kontekst
    "embeddings_model": "all-MiniLM-L6-v2",   # Model til at konvertere tekst til vektorer (384D)
    "llm_model": "qwen2.5:3b",                # LLM-model i Ollama (3 milliarder parametre)
    "chroma_db_path": "squad_rag_db",         # Sti til Chroma DB-mappe (persistent lagring)
    "ollama_num_predict": 128,               # Maksimal længde af LLM-svar (tokens)
}

# --- Hjælpefunktioner ---
def download_squad() -> None:
    """
    Downloader SQuAD-dataset fra GitHub, hvis det ikke allerede eksisterer lokalt.

    Args:
        None

    Returns:
        None

    Side Effects:
        Downloader filen `train-v1.1.json` til den aktuelle mappe, hvis den ikke findes.
    """
    if not os.path.exists(CONFIG["squad_file"]):
        print("Downloader SQuAD-dataset (120MB)...")
        os.system(f"wget -q https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/{CONFIG['squad_file']}")
        print(f"Download færdig: {CONFIG['squad_file']}")
    else:
        print(f"SQuAD-fil findes allerede: {CONFIG['squad_file']}")

def load_squad_subset(max_questions: int = CONFIG["max_questions"]) -> Tuple[List[str], List[str]]:
    """
    Indlæser et subset af SQuAD-dataset ved hjælp af streaming-parsing (sparer RAM).
    Bruger ijson til at parse JSON-filen i realtid uden at indlæse hele filen i hukommelsen.

    Args:
        max_questions: Maksimal antal spørgsmål at indlæse (default: 1000).

    Returns:
        Tuple[List[str], List[str]]:
            - contexts: Liste af kontekst-strenge (tekster fra SQuAD).
            - questions: Liste af spørgsmål (fra SQuAD).

    Example:
        >>> contexts, questions = load_squad_subset(max_questions=100)
        >>> len(questions)
        100
    """
    contexts = []
    questions = []
    question_count = 0
    current_context = None

    print(f"Indlæser {max_questions} spørgsmål fra SQuAD...")

    with open(CONFIG["squad_file"], "rb") as f:
        parser = ijson.parse(f)
        for prefix, event, value in parser:
            if prefix == "data.item.paragraphs.item.context":
                current_context = value
            elif prefix == "data.item.paragraphs.item.qas.item.question":
                if question_count < max_questions and current_context:
                    questions.append(value)
                    contexts.append(current_context)
                    question_count += 1
                if question_count >= max_questions:
                    break
    print(f"Indlæst {len(questions)} spørgsmål fra {len(set(contexts))} unikke kontekster.")
    return contexts, questions

def clean_text(text: str) -> str:
    """
    Renser en tekststreng ved at fjerne ekstra mellemrum og specialtegn.
    Bevarer kun alphanumeriske tegn, mellemrum, og punktering (.,!?;:).

    Args:
        text: Den tekst, der skal renses.

    Returns:
        str: Den rensede tekst.

    Example:
        >>> clean_text("  Dette   er  en   test!  ")
        "Dette er en test!"
    """
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,!?;:]', '', text)
    return text.strip()

def clean_and_chunk(texts: List[str], chunk_size: int = CONFIG["chunk_size"], chunk_overlap: int = CONFIG["chunk_overlap"]) -> List[str]:
    """
    Renser og opdeler en liste af tekststrenge i mindre chunks.
    Bruger RecursiveCharacterTextSplitter til at opdele tekster på en intelligent måde,
    der bevarer ord og sætninger.

    Args:
        texts: Liste af tekststrenge, der skal renses og opdeles.
        chunk_size: Størrelsen på hver chunk i tegn (default: 500).
        chunk_overlap: Antal tegn, der overlapper mellem chunks (default: 50).

    Returns:
        List[str]: Liste af rensede og opdelte tekst-chunks.

    Example:
        >>> clean_and_chunk(["Dette er en lang tekst..."], chunk_size=20, chunk_overlap=5)
        ["Dette er en lang", "en lang tekst..."]
    """
    cleaned_texts = [clean_text(text) for text in texts]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = []
    for text in cleaned_texts:
        chunks.extend(text_splitter.split_text(text))
    print(f"Oprettet {len(chunks)} chunks (størrelse: {chunk_size} tegn, overlap: {chunk_overlap}).")
    return chunks

# --- Chroma DB Opsætning ---
def setup_chroma_db(chunks: List[str]) -> chromadb.Collection:
    """
    Opretter en Chroma DB-kollektion og gemmer chunks med deres embeddings.
    Chroma DB bruges til hurtigt at søge i vektorer ved hjælp af cosinus-lighed.

    Args:
        chunks: Liste af tekst-chunks, der skal gemmes i databasen.

    Returns:
        chromadb.Collection: Den oprettede Chroma DB-kollektion med chunks og embeddings.

    Side Effects:
        Opretter en persistent Chroma DB i mappen `squad_rag_db`.
    """
    print("Opretter Chroma DB...")
    client = chromadb.PersistentClient(path=CONFIG["chroma_db_path"])
    collection = client.get_or_create_collection(name="squad_contexts")

    print(f"Indlæser embeddings-model: {CONFIG['embeddings_model']}...")
    model = SentenceTransformer(CONFIG["embeddings_model"])

    print("Generer embeddings...")
    embeddings = model.encode(chunks, batch_size=32).tolist()

    ids = [f"id_{i}" for i in range(len(chunks))]
    collection.add(documents=chunks, embeddings=embeddings, ids=ids)
    print(f"Gemt {len(chunks)} chunks i Chroma DB ({CONFIG['chroma_db_path']}).")
    return collection, model

# --- RAG-System ---
def ask_rag(collection: chromadb.Collection, model: SentenceTransformer, query: str, n_results: int = 3) -> str:
    """
    Udfører en RAG-forespørgsel: Søger i Chroma DB og generer et svar ved hjælp af LLM'en.

    Args:
        collection: Chroma DB-kollektion med gemte chunks og embeddings.
        model: SentenceTransformer-model til at generere embeddings for spørgsmålet.
        query: Spørgsmålet, der skal besvares.
        n_results: Antal chunks at returnere fra Chroma DB (default: 3).

    Returns:
        str: LLM'ens genererede svar baseret på den fundne kontekst.

    Example:
        >>> ask_rag(collection, model, "Hvornår begyndte Scholastic Magazine?")
        "Scholastic Magazine begyndte at blive udgivet i september 1876."
    """
    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )
    context = "\n\n".join(results["documents"][0])

    ollama_client = Client()
    prompt = f"""
    Besvar spørgsmålet baseret på konteksten nedenfor.
    Hvis svaret ikke findes i konteksten, sig "Jeg ved det ikke".

    Kontekst:
    {context}

    Spørgsmål: {query}
    Svar:
    """

    response = ollama_client.generate(
        model=CONFIG["llm_model"],
        prompt=prompt,
        options={"num_predict": CONFIG["ollama_num_predict"]}
    )
    return response["response"]

# --- Evaluér RAG-System ---
def evaluate_rag(collection: chromadb.Collection, model: SentenceTransformer, questions: List[str], answers: List[str], num_questions: int = 10) -> float:
    """
    Evaluere RAG-systemet ved at sammenligne LLM-svar med korrekte svar fra SQuAD.

    Args:
        collection: Chroma DB-kollektion med gemte chunks og embeddings.
        model: SentenceTransformer-model til at generere embeddings.
        questions: Liste af spørgsmål at evaluere på.
        answers: Liste af korrekte svar (svarende til spørgsmålene).
        num_questions: Antal spørgsmål at evaluere (default: 10).

    Returns:
        float: Præcision som procentdel (f.eks. 50.0 for 5/10 korrekte).

    Side Effects:
        Printer detaljerede resultater for hver spørgsmål (OK/ERR).

    Example:
        >>> accuracy = evaluate_rag(collection, model, test_questions, test_answers, num_questions=5)
        Præcision: 80.0% (4/5 korrekte)
    """
    correct = 0
    for i in range(min(num_questions, len(questions))):
        query = questions[i]
        true_answer = answers[i].lower()
        rag_answer = ask_rag(collection, model, query).lower()

        if true_answer in rag_answer:
            correct += 1
            status = "[OK]"
        else:
            status = "[ERR]"

        print(f"\n{status} Spørgsmål: {query}")
        print(f"   Korrekt svar: {true_answer}")
        print(f"   RAG-svar: {rag_answer}")

    accuracy = (correct / num_questions) * 100
    print(f"\nPræcision: {accuracy:.1f}% ({correct}/{num_questions} korrekte)")
    return accuracy

# --- Hovedprogram ---
def main():
    """
    Hovedfunktion, der kører hele RAG-workflow:
    1. Downloader SQuAD-dataset (hvis nødvendigt).
    2. Indlæser og opdeler data i chunks.
    3. Opretter Chroma DB med embeddings.
    4. Tester systemet med 10 spørgsmål.
    5. Åbner interaktivt spørgsmålsinterface.

    Args:
        None

    Returns:
        None
    """
    print("=" * 60)
    print("SQuAD RAG-System")
    print("=" * 60)

    download_squad()
    contexts, questions = load_squad_subset(CONFIG["max_questions"])
    chunks = clean_and_chunk(contexts)
    collection, embeddings_model = setup_chroma_db(chunks)

    print("\n" + "=" * 60)
    print("Test af RAG-systemet")
    print("=" * 60)

    test_questions = questions[:10]
    test_answers = []
    with open(CONFIG["squad_file"], "rb") as f:
        parser = ijson.parse(f)
        answer_count = 0
        for prefix, event, value in parser:
            if prefix == "data.item.paragraphs.item.qas.item.answers.item.text" and answer_count < 10:
                test_answers.append(value)
                answer_count += 1

    evaluate_rag(collection, embeddings_model, test_questions, test_answers, num_questions=10)

    print("\n" + "=" * 60)
    print("Interaktivt RAG-system (tryk Ctrl+C for at stoppe)")
    print("=" * 60)

    while True:
        try:
            query = input("\nIndtast dit spørgsmål (eller 'exit'): ").strip()
            if query.lower() in ("exit", "quit", "q"):
                break
            answer = ask_rag(collection, embeddings_model, query)
            print(f"\nSvar: {answer}")
        except KeyboardInterrupt:
            print("\nFarvel!")
            break
        except Exception as e:
            print(f"[ERR] Fejl: {e}")

if __name__ == "__main__":
    main()