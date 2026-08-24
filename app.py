import spaces
import os
import gradio as gr
import torch
from sentence_transformers import SentenceTransformer
import re
import numpy as np
import faiss

import pymupdf4llm

# ---------------------------------------------------------
# PDF PATH
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PDF_FILE = os.path.join(BASE_DIR,"Earth Our Planet data from web.pdf")

if not os.path.exists(PDF_FILE):
    raise FileNotFoundError(
        f"PDF not found: {PDF_FILE}"
    )


# ---------------------------------------------------------
# READ PDF
# ---------------------------------------------------------

text = pymupdf4llm.to_markdown(PDF_FILE)

print("Extracted characters:", len(text))

# ---------------------------------------------------------
#CHUNKING
# ---------------------------------------------------------

def chunk_document(
    text: str,
    source: str,
    chunk_size: int = 1200,
    overlap: int = 200
):

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size"
        )

  # ---------------------------------------------------------
  # Clean text
  # ---------------------------------------------------------

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Normalize spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    text = text.strip()

    if not text:
        raise ValueError("No text was extracted from the PDF.")

# ---------------------------------------------------------
# Identify Markdown headings
# ---------------------------------------------------------

    lines = text.split("\n")

    current_chapter = None
    current_section = None
    current_subsection = None

    sections = []
    current_content = []

# ---------------------------------------------------------
# Save current section
# ---------------------------------------------------------

    def save_section():

        if current_content:

            content = "\n".join(
                current_content
            ).strip()

            if content:

                sections.append({
                    "chapter": current_chapter,
                    "section": current_section,
                    "subsection": current_subsection,
                    "text": content
                })

# ---------------------------------------------------------
# Build hierarchical sections
# ---------------------------------------------------------

    for line in lines:

        line = line.strip()

        # Blank line
        if not line:

            if (
                current_content
                and current_content[-1] != ""
            ):
                current_content.append("")

            continue

        # -----------------------------------------------------
        # H1 = Chapter
        # -----------------------------------------------------

        if re.match(r"^#\s+", line):

            save_section()

            current_content = []

            current_chapter = re.sub(
                r"^#\s+",
                "",
                line
            )

            current_section = None
            current_subsection = None

            current_content.append(line)

        # -----------------------------------------------------
        # H2 = Section
        # -----------------------------------------------------

        elif re.match(r"^##\s+", line):

            save_section()

            current_content = []

            current_section = re.sub(
                r"^##\s+",
                "",
                line
            )

            current_subsection = None

            current_content.append(line)

        # -----------------------------------------------------
        # H3 = Subsection
        # -----------------------------------------------------

        elif re.match(r"^###\s+", line):

            save_section()

            current_content = []

            current_subsection = re.sub(
                r"^###\s+",
                "",
                line
            )

            current_content.append(line)

        # -----------------------------------------------------
        # Normal content
        # -----------------------------------------------------

        else:

            current_content.append(line)

    # Save final section
    save_section()

# ---------------------------------------------------------
# Split sections into chunks
# ---------------------------------------------------------

    documents = []

    chunk_counter = 0

    for section in sections:

        section_text = section["text"]

# -----------------------------------------------------
# Section already fits
# -----------------------------------------------------

        if len(section_text) <= chunk_size:

            documents.append({

                "id": f"chunk_{chunk_counter}",

                "source": source,

                "chunk_index": chunk_counter,

                "chapter": section["chapter"],

                "section": section["section"],

                "subsection": section["subsection"],

                "text": section_text

            })

            chunk_counter += 1

            continue

        # -----------------------------------------------------
        # Large section
        # -----------------------------------------------------

        paragraphs = re.split(
            r"\n\s*\n",
            section_text
        )

        current_chunk = ""

        for paragraph in paragraphs:

            paragraph = paragraph.strip()

            if not paragraph:
                continue

            # -------------------------------------------------
            # Paragraph fits current chunk
            # -------------------------------------------------

            if (
                len(current_chunk)
                + len(paragraph)
                + 2
                <= chunk_size
            ):

                if current_chunk:
                    current_chunk += "\n\n"

                current_chunk += paragraph

            # -------------------------------------------------
            # Paragraph doesn't fit
            # -------------------------------------------------

            else:

                if current_chunk:

                    documents.append({

                        "id": f"chunk_{chunk_counter}",

                        "source": source,

                        "chunk_index": chunk_counter,

                        "chapter": section["chapter"],

                        "section": section["section"],

                        "subsection": section["subsection"],

                        "text": current_chunk.strip()

                    })

                    chunk_counter += 1

                # -------------------------------------------------
                # Very large paragraph
                # -------------------------------------------------

                if len(paragraph) > chunk_size:

                    start = 0

                    while start < len(paragraph):

                        end = start + chunk_size

                        small_chunk = paragraph[start:end].strip()

                        if small_chunk:

                            documents.append({

                                "id": f"chunk_{chunk_counter}",

                                "source": source,

                                "chunk_index": chunk_counter,

                                "chapter": section["chapter"],

                                "section": section["section"],

                                "subsection": section["subsection"],

                                "text": small_chunk

                            })

                            chunk_counter += 1

                        start = end - overlap

                    current_chunk = ""

                else:

                    current_chunk = paragraph

        # -----------------------------------------------------
        # Save final chunk
        # -----------------------------------------------------

        if current_chunk:

            documents.append({

                "id": f"chunk_{chunk_counter}",

                "source": source,

                "chunk_index": chunk_counter,

                "chapter": section["chapter"],

                "section": section["section"],

                "subsection": section["subsection"],

                "text": current_chunk.strip()

            })

            chunk_counter += 1

    # ---------------------------------------------------------
    # Add previous / next relationships
    # ---------------------------------------------------------

    for i, document in enumerate(documents):

        document["previous_chunk"] = (
            documents[i - 1]["id"]
            if i > 0
            else None
        )

        document["next_chunk"] = (
            documents[i + 1]["id"]
            if i < len(documents) - 1
            else None
        )

    return documents


documents = chunk_document(text=text,source='Earth Our Planet data from web.pdf',chunk_size=1200,overlap=200)


import spaces

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

texts = [
    document["text"]
    for document in documents
]


@spaces.GPU
def generate_embeddings(texts):
    return model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True
    )


embeddings = generate_embeddings(texts)

print("Embedding shape:", embeddings.shape)

# =========================================================
# CREATE FAISS SEARCH INDEX
# =========================================================


# Convert embeddings to NumPy float32
embedding_matrix = np.asarray(embeddings).astype("float32")



# Get embedding dimension
dimension = embedding_matrix.shape[1]

index = faiss.IndexFlatIP(dimension)


# Add embeddings
index.add(embedding_matrix)

print("Number of vectors:", index.ntotal)
print("Embedding dimension:", dimension)


##CREATE SEARCH
def search_documents(query, top_k=5):
# Create embedding for the user's question
    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32"
    )

    # Search FAISS
    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        # Ignore invalid FAISS results
        if idx < 0:
            continue

        result = documents[idx].copy()

        result["score"] = float(score)

        results.append(result)

    return results


####------------------------------------------
###Test retrieval
####------------------------------------------


query = "What are the main features of Earth?"

results = search_documents(
    query,
    top_k=5
)

for result in results:

    print("\n" + "=" * 80)

    print("Score:", result["score"])
    print("Chunk:", result["chunk_index"])
    print("Chapter:", result["chapter"])
    print("Section:", result["section"])

    print("\nTEXT:")
    print(result["text"])


    ##Build the context for the LLM

def build_context(results):

    context_parts = []

    for result in results:

        context_parts.append(
            f"""---- Document Chunk {result['chunk_index']} ---
Chapter: {result['chapter']}
Section: {result['section']}

{result['text']}"""
        )

    return "\n".join(context_parts)


####------------------------------------------
#### RETRIEVAL FROM SOURCES
####------------------------------------------


def retrieve_context(query, top_k=5):

    results = search_documents(query,top_k)

    context = build_context(results)

    sources = []

    for result in results:

        sources.append({

            "source": result["source"],

            "chunk": result["chunk_index"],

            "score": result["score"],

            "chapter": result["chapter"],

            "section": result["section"]

        })

    return context, sources




    ##TEST

query = "What is Earth made of?"

context, sources = retrieve_context(query,top_k=5)

print("CONTEXT")
print("=" * 80)

print(context)

print("\nSOURCES")
print("=" * 80)

for source in sources:
    print(source)


import gradio as gr


def chatbot(message, history):

    try:

        results = search_documents(
            message,
            top_k=1
        )

        if not results:

            return "I could not find relevant information in the document."

        response = "## Relevant information\n\n"

        for i, result in enumerate(
            results,
            start=1
        ):

            response += (
                f"### Result {i}\n\n"
                f"**Similarity:** "
                f"{result['score']:.3f}\n\n"
                f"**Chapter:** "
                f"{result['chapter']}\n\n"
                f"**Section:** "
                f"{result['section']}\n\n"
                f"{result['text']}\n\n"
                f"---\n\n"
            )

        return response

    except Exception as e:

        return (
            f"Error: {type(e).__name__}: {e}"
        )


# ---------------------------------------------------------
# Gradio
# ---------------------------------------------------------

def chatbot(message, history):

    try:

        print("=" * 60)
        print("USER QUESTION:", message)

        results = search_documents(
            message,
            top_k=3
        )

        print("NUMBER OF RESULTS:", len(results))

        if not results:
            print("NO RESULTS")
            return "No results found."

        for i, result in enumerate(results, 1):

            print(
                f"RESULT {i} SCORE:",
                result["score"]
            )

            print(
                "TEXT:",
                result["text"][:300]
            )

        # DO NOT reject based on 0.30 yet

        result = results[0]

        return (
            f"### Retrieved Answer\n\n"
            f"{result['text']}\n\n"
            f"---\n\n"
            f"**Similarity:** "
            f"{result['score']:.3f}"
        )

    except Exception as e:

        print("ERROR:", type(e).__name__, str(e))

        return (
            f"Error: {type(e).__name__}: {e}"
        )
# ---------------------------------------------------------
# Launch
# ---------------------------------------------------------

demo = gr.ChatInterface(
    fn=chatbot,
    title="AI Document Intelligence",
    description=(
        "Ask questions about the Earth document. "
        "The system retrieves relevant information "
        "using semantic search."
    )
)


if __name__ == "__main__":
    demo.launch()
