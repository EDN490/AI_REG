"""
SQuAD RAG-System
================
Sidst opdateret: 2026-09-22 07:07:15

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
HVORDAN RAG FUNGERER
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

RAG (Retrieval-Augmented Generation) er en arkitektur, der kombinerer:

1. **Retrieval**: Søgning i en vektordatabase (Chroma DB) for at finde relevant kontekst.
2. **Augmented Generation**: LLM'en (Qwen2.5-3B) generer svar baseret på den fundne kontekst.

Workflow:
   [Spørgsmål] → [Konverter til vektor] → [Søg i Chroma DB] → [Top-N chunks]
        ↓
   [Send chunks til LLM] → [Generer svar baseret på kontekst]

Fordele ved RAG:

- **Reducerer hallucinationer**: LLM'en instrueres i kun at svare ud fra konteksten, ikke sin træning.
- **Aktuelle oplysninger**: Kan svare baseret på specifikke data (f.eks. SQuAD-dataset).
- **Transparens**: Svaret kan spores tilbage til konteksten.

Begrænsninger:

- **Afhængig af kontekstkvalitet**: Hvis konteksten mangler svaret, kan LLM'en ikke svare korrekt.
- **Chunking-problemer**: For små chunks kan splitte vigtige oplysninger.
- **Embeddings-præcision**: Dårlige embeddings giver dårlige matches.

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
SQuAD-DATASET
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
Direkte download (JSON):
   https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/master/dataset/train-v1.1.json

Officiel hjemmeside:
   https://rajpurkar.github.io/SQuAD-explorer/

Hvad er SQuAD?

   SQuAD (Stanford Question Answering Dataset) er et dataset med 100.000+
   spørgsmål-svar-par, baseret på Wikipedia-artikler. Hvert spørgsmål har et
   korrekt svar, der er et direkte uddrag fra en kontekst-tekst. Datasetet
   bruges til at træne og teste modeller i tekstforståelse (reading comprehension).

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
CENTRALE KONCEPTER
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

Embeddings:

   Konverterer tekst til numeriske vektorer (talrækker), så computeren kan
   forstå tekstens betydning og sammenligne tekster. I dette program bruger
   vi modellen 'all-MiniLM-L6-v2' (384 dimensioner, 80MB) til at omdanne SQuAD-tekster.
   Andre populære modeller: 'all-mpnet-base-v2' (768 dimensioner, 420MB).

Chroma DB:

   En vektordatabase, der effektivt gemmer og søger i embeddings. Når du
   stiller et spørgsmål, konverteres det til en vektor, og Chroma DB finder
   hurtigt de mest lignende vektorer. I dette program er databasen sat op til
   cosinus-afstand (hnsw:space = cosine).

   Cosinus-lighed måler vinklen mellem vektorer:
   1.0 = identisk retning (perfekt match),
   0.0 = ingen lighed,
   -1.0 = modsat retning.

   Chroma returnerer cosinus-AFSTAND, som er 1 - cosinus-lighed:
   0.0 = identisk retning, lavere er bedre.

LLM/Ollama:

   LLM (Large Language Model) er en AI-model, der kan forstå og generere
   menneskelig tekst. Ollama er en lokal server, der gør det nemt at køre
   LLM'er på din egen maskine. I dette program bruger vi Qwen2.5-3B til at
   generere svar baseret på den kontekst, der er fundet via Chroma DB.

   Data, embeddings-model og prompt er alle på engelsk. Stil derfor også
   spørgsmål på engelsk i den interaktive del.

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
KOMPONENTER OG DERES VIRKEN
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

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

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
DESIGN-BESKRIVELSE
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

Arkitektur:
  SQuAD-dataset (JSON)
       ↓
  [Load Subset] -> Indlæser 5.000 spørgsmål + kontekster
       ↓
  [Clean & Chunk] -> Opdeler unikke kontekster i 800-tegn chunks
       ↓
  [Embeddings] -> Konverterer chunks til vektorer (all-MiniLM-L6-v2)
       ↓
  [Chroma DB] -> Gemmer vektorer i lokal database
       ↓
  [RAG Query] -> Søger i Chroma DB + sender top-5 chunks til LLM
       ↓
  [Ollama (Qwen2.5-3B)] -> Generer svar baseret på kontekst
       ↓
  [Output] -> Returnerer svar til brugeren

Workflow:
  1. Forberedelse: Indlæs SQuAD -> Fjern dublerede kontekster -> Chunk -> Embed -> Gem i Chroma DB
  2. Spørgsmål: Søg i Chroma DB -> Send top-5 chunks -> Få svar fra LLM
  3. Evaluering: Sammenlign LLM-svar med korrekte svar fra SQuAD

Optimeringer:

  - Streaming JSON-parsing (ijson) for at spare RAM
  - Batch-processing af embeddings (32 chunks ad gangen)
  - Unikke SQuAD-kontekster chunkes kun én gang
  - Test-databasen slettes og opbygges fra bunden ved hver kørsel

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
EKSEMPLER
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

Når systemet køres, kan du stille spørgsmål som:

- "To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?"
  → Svar: "Saint Bernadette Soubirous"

- "How many student newspapers are found at Notre Dame?"
  → Svar: "Three"

- "When did the Scholastic Magazine of Notre Dame begin publishing?"
  → Svar: "September 1876"

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
FEJLSØGNING
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

- [ModuleNotFoundError]:
  - Sikre, at alle pakker er installeret:
    `pip install ollama chromadb sentence-transformers ijson langchain-text-splitters`.

- [Chroma DB returnerer ingen resultater]:
  - Test-databasen oprettes automatisk fra bunden ved hver kørsel.

- [LLM'en siger "I don't know" for ofte]:
  - Vi bruger `n_results=5` for at give LLM'en mere relevant kontekst.
  - Vi bruger 800-tegn chunks med 100 tegn overlap.

- [Forkert svar]:
  - Tjek om konteksten indeholder det korrekte svar. Hvis ja, kan fejlen
    ligge i LLM'ens valg af svar.

Output-format under evaluering:
- [OK]: RAG-svaret matchede det korrekte svar (uanset case).
- [ERR]: RAG-svaret var forkert eller ufuldstændigt.
- Præcision: Procentdel af korrekte svar (f.eks. 90.0% = 9/10 korrekte).

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
ANVENDELSE:
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

- Kundesupport: Automatiser FAQ-svar baseret på produktdokumentation.
- Intern videnbase: Gør medarbejdere i stand til hurtigt at finde HR/IT-oplysninger.
- Faglig støtte: Hjælp eksperter (læger, advokater) med at finde præcise oplysninger i store dokumenter.

::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::
KORT INSTALLATIONSVEJLEDNING
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

1. Installer Ollama:
   curl -fsSL https://ollama.ai/install.sh | sh

2. Træk model:
   ollama pull qwen2.5:3b

3. Opret virtuelt miljø (anbefalet):
   python -m venv .venv

4. Aktivér det:
   - Linux/macOS: source .venv/bin/activate
   - Windows: .venv\\Scripts\\activate

5. Installer Python-afhængigheder:
   pip install ollama chromadb sentence-transformers ijson langchain-text-splitters

6. Kør:
   python3 squad_rag.py

Krav: Linux, macOS, eller Windows (med WSL) + Python 3.8+
Testet på: Raspberry Pi 5 (8GB RAM) med Ubuntu 24.04
"""

import os
import re
import shutil
import ijson
from typing import List, Tuple

try:
    from ollama import Client
    from sentence_transformers import SentenceTransformer
    import chromadb
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError as e:
    print(f"[ERR] Manglende bibliotek: {e}")
    print(
        "Installer med: pip install ollama chromadb "
        "sentence-transformers ijson langchain-text-splitters"
    )
    exit(1)


# --- Konfiguration ---
CONFIG = {
    # ------------------------------------------------------------------
    # Datasæt
    # ------------------------------------------------------------------

    # Filnavn på SQuAD-træningsdatasættet (JSON).
    # Downloades automatisk af download_squad(), hvis filen mangler.
    "squad_file": "train-v1.1.json",

    # Hvor mange spørgsmål (med tilhørende facit og kontekst) der indlæses.
    # Datasættet læses fra starten, så det er de første spørgsmål/artikler,
    # der bruges - ikke en tilfældig stikprøve. Sat op fra 1000 til 5000
    # for at få flere unikke kontekster med i databasen.
    "max_questions": 5000,

    # ------------------------------------------------------------------
    # Chunking (opdeling af tekst)
    # ------------------------------------------------------------------

    # Maksimal længde af en chunk, målt i TEGN (ikke tokens).
    # Oprindeligt ændret ved en test på 10 spørgsmål, der gav 90 %.
    # Bemærk: 10 spørgsmål er for få til at konkludere noget. Mange
    # SQuAD-afsnit er ca. 700-800 tegn, så de fleste bliver kun til én chunk.
    "chunk_size": 800,

    # Antal tegn, der gentages mellem to nabo-chunks, så en sætning på
    # grænsen ikke mister sin sammenhæng.
    "chunk_overlap": 100,

    # ------------------------------------------------------------------
    # Modeller
    # ------------------------------------------------------------------

    # Sentence Transformers-model, der laver tekst om til vektorer
    # (384 dimensioner). Modellen er engelsk, så både chunks og spørgsmål
    # bør være på engelsk.
    "embeddings_model": "all-MiniLM-L6-v2",

    # Navn på LLM'en i Ollama. Skal være hentet på forhånd med:
    #   ollama pull qwen2.5:3b
    "llm_model": "qwen2.5:3b",

    # ------------------------------------------------------------------
    # Chroma DB
    # ------------------------------------------------------------------

    # Mappen hvor Chroma gemmer databasen på disken.
    # Separat test-database: den slettes og opbygges fra bunden ved hver
    # kørsel (se setup_chroma_db), så den kan ikke blandes med en
    # eventuel rigtig database.
    "chroma_db_path": "squad_rag_test_db",

    # Navn på samlingen (collection) i databasen - kan sammenlignes med
    # en tabel, der indeholder alle chunks og deres vektorer.
    "chroma_collection": "squad_contexts_test",

    # ------------------------------------------------------------------
    # Retrieval og generering
    # ------------------------------------------------------------------

    # Antal chunks, der hentes fra Chroma pr. spørgsmål og sendes til LLM'en.
    # Ændret fra 3 til 5 for at give LLM'en mere kontekst. Flere chunks
    # giver større chance for, at svaret er med, men også en længere
    # prompt og langsommere svar.
    "n_results": 5,

    # Maksimalt antal tokens, LLM'en må generere i sit svar.
    # 128 er rigeligt til korte svar; længere svar afkortes.
    "ollama_num_predict": 128,
}


# --- Hjælpefunktioner ---
def download_squad() -> None:
    """
    Downloader SQuAD-dataset fra GitHub, hvis det ikke allerede eksisterer lokalt.
    """
    if not os.path.exists(CONFIG["squad_file"]):
        print("Downloader SQuAD-dataset (120MB)...")
        os.system(
            f"wget -q https://raw.githubusercontent.com/rajpurkar/"
            f"SQuAD-explorer/master/dataset/{CONFIG['squad_file']}"
        )
        print(f"Download færdig: {CONFIG['squad_file']}")
    else:
        print(f"SQuAD-fil findes allerede: {CONFIG['squad_file']}")


def load_squad_subset(
    max_questions: int = CONFIG["max_questions"]
) -> Tuple[List[str], List[str], List[str]]:
    """
    Indlæser et subset af SQuAD-dataset.

    Spørgsmål, korrekte svar og kontekst gemmes samtidig,
    så de altid forbliver korrekt parret.
    """
    contexts = []
    questions = []
    answers = []

    print(f"Indlæser {max_questions} spørgsmål fra SQuAD...")
    with open(CONFIG["squad_file"], "rb") as f:
        for item in ijson.items(f, "data.item"):
            for paragraph in item["paragraphs"]:
                context = paragraph["context"]

                for qa in paragraph["qas"]:
                    if len(questions) >= max_questions:
                        break

                    questions.append(qa["question"])
                    answers.append(qa["answers"][0]["text"])
                    contexts.append(context)

                if len(questions) >= max_questions:
                    break

            if len(questions) >= max_questions:
                break

    print(
        f"Indlæst {len(questions)} spørgsmål fra "
        f"{len(set(contexts))} unikke kontekster."
    )

    return contexts, questions, answers


def clean_text(text: str) -> str:
    """
    Normaliserer mellemrum (flere mellemrum/linjeskift bliver til ét).

    Tegnsætning og specialtegn bevares bevidst. SQuAD-tekst er allerede ren,
    og fjernelse af tegn som bindestreger, apostroffer og procenttegn
    ændrer teksten, så facit ikke længere kan matches.
    """
    return re.sub(r"\s+", " ", text).strip()


def clean_and_chunk(
    texts: List[str],
    chunk_size: int = CONFIG["chunk_size"],
    chunk_overlap: int = CONFIG["chunk_overlap"]
) -> List[str]:
    """
    Renser og opdeler tekst i chunks.

    Identiske kontekster fjernes først, så den samme kontekst
    ikke bliver chunket flere gange.
    """

    unique_texts = list(
        dict.fromkeys(texts)
    )

    print(
        f"Chunker {len(unique_texts)} unikke contexts "
        f"ud af {len(texts)} contexts."
    )

    cleaned_texts = [
        clean_text(text)
        for text in unique_texts
    ]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )

    chunks = []

    for text in cleaned_texts:
        chunks.extend(
            text_splitter.split_text(text)
        )

    print(
        f"Oprettet {len(chunks)} chunks "
        f"(størrelse: {chunk_size} tegn, "
        f"overlap: {chunk_overlap})."
    )

    return chunks


# --- Chroma DB Opsætning ---
def setup_chroma_db(
    chunks: List[str]
) -> Tuple[chromadb.Collection, SentenceTransformer]:
    """
    Opretter en ny test-DB fra bunden.
    """
    print("Opretter ny Chroma DB...")

    shutil.rmtree(
        CONFIG["chroma_db_path"],
        ignore_errors=True
    )

    client = chromadb.PersistentClient(
        path=CONFIG["chroma_db_path"]
    )

    # Cosinus-afstand i stedet for Chromas standard (kvadreret L2).
    collection = client.get_or_create_collection(
        name=CONFIG["chroma_collection"],
        metadata={"hnsw:space": "cosine"},
    )

    print(
        f"Indlæser embeddings-model: "
        f"{CONFIG['embeddings_model']}..."
    )

    model = SentenceTransformer(
        CONFIG["embeddings_model"]
    )

    print("Genererer embeddings...")

    embeddings = model.encode(
        chunks,
        batch_size=32
    ).tolist()

    ids = [
        f"id_{i}"
        for i in range(len(chunks))
    ]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids
    )

    print(
        f"Gemt {len(chunks)} chunks i test-DB "
        f"({CONFIG['chroma_db_path']})."
    )

    return collection, model


# --- RAG-System ---
def ask_rag(
    collection: chromadb.Collection,
    model: SentenceTransformer,
    ollama_client: Client,
    query: str,
    n_results: int = CONFIG["n_results"]
) -> str:
    """
    Udfører en RAG-forespørgsel.
    """

    query_embedding = model.encode(
        [query]
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        include=[
            "documents",
            "distances"
        ]
    )

    documents = results["documents"][0]
    distances = results["distances"][0]

    # Fjern identiske chunks fra retrieval-resultatet.
    unique_documents = []
    unique_distances = []
    seen = set()

    for doc, distance in zip(
        documents,
        distances
    ):
        if doc in seen:
            continue

        seen.add(doc)

        unique_documents.append(doc)
        unique_distances.append(distance)

    print(
        f"\n   Retrieval: "
        f"{len(documents)} resultater -> "
        f"{len(unique_documents)} unikke chunks"
    )

    for idx, (doc, distance) in enumerate(
        zip(unique_documents, unique_distances),
        1
    ):
        print(
            f"\n   Chunk {idx} "
            f"(cosinus-afstand: {distance:.4f}):"
        )
        print(
            f"   {doc}"
        )

    context = "\n\n".join(
        unique_documents
    )

    # query er brugerens spørgsmål.
    # context indeholder de relevante originale tekststumper (chunks) fra
    # SQuAD-datasættet, som er fundet via ChromaDB og bruges som kontekst
    # til at besvare spørgsmålet.
    # Prompten er på engelsk, fordi data og embeddings-model er engelske.
    prompt = (
        "Answer the question briefly and precisely, based only on the "
        "context below.\n"
        "Do not use any knowledge that is not in the context.\n"
        "If the answer is not in the context, say \"I don't know\".\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    response = ollama_client.generate(
        model=CONFIG["llm_model"],
        prompt=prompt,
        options={
            "num_predict": CONFIG["ollama_num_predict"]
        }
    )

    return response["response"]


# --- Evaluér RAG-System ---
def evaluate_rag(
    collection: chromadb.Collection,
    model: SentenceTransformer,
    ollama_client: Client,
    questions: List[str],
    answers: List[str],
    num_questions: int = 50
) -> float:
    """
    Evaluerer RAG-systemet ved at sammenligne LLM-svar
    med korrekte SQuAD-svar.
    """

    correct = 0

    questions_to_evaluate = min(
        num_questions,
        len(questions),
        len(answers)
    )

    for i in range(questions_to_evaluate):
        query = questions[i]
        true_answer = answers[i].lower()

        rag_answer = ask_rag(
            collection,
            model,
            ollama_client,
            query
        ).lower()

        if true_answer in rag_answer:
            correct += 1
            status = "[OK]"
        else:
            status = "[ERR]"

        print(
            f"\n{status} Spørgsmål: {query}"
        )

        print(
            f"   Korrekt svar: {true_answer}"
        )

        print(
            f"   RAG-svar: {rag_answer}"
        )

    if questions_to_evaluate == 0:
        print(
            "\nIngen spørgsmål kunne evalueres."
        )
        return 0.0

    accuracy = (
        correct / questions_to_evaluate
    ) * 100

    print(
        f"\nPræcision: {accuracy:.1f}% "
        f"({correct}/{questions_to_evaluate} korrekte)"
    )

    return accuracy


# --- Hovedprogram ---
def main():
    """
    Hovedfunktion for hele RAG-workflowet.
    """

    print("=" * 60)
    print("SQuAD RAG-System")
    print("=" * 60)

    download_squad()

    contexts, questions, answers = load_squad_subset(
        CONFIG["max_questions"]
    )

    chunks = clean_and_chunk(
        contexts
    )

    collection, embeddings_model = setup_chroma_db(
        chunks
    )

    # Genbrug én Ollama-client.
    ollama_client = Client()

    print("\n" + "=" * 60)
    print("Test af RAG-systemet")
    print("=" * 60)

    evaluate_rag(
        collection,
        embeddings_model,
        ollama_client,
        questions,
        answers,
        num_questions=50
    )

    print("\n" + "=" * 60)
    print(
        "Interaktivt RAG-system "
        "(tryk Ctrl+C for at stoppe)"
    )
    print("=" * 60)

    while True:
        try:
            query = input(
                "\nIndtast dit spørgsmål på engelsk (eller 'exit'): "
            ).strip()

            if query.lower() in (
                "exit",
                "quit",
                "q"
            ):
                break

            answer = ask_rag(
                collection,
                embeddings_model,
                ollama_client,
                query
            )

            print(
                f"\nSvar: {answer}"
            )

        except KeyboardInterrupt:
            print("\nFarvel!")
            break

        except Exception as e:
            print(
                f"[ERR] Fejl: {e}"
            )


if __name__ == "__main__":
    main()
