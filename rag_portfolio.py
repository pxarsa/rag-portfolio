
import numpy as np
import faiss
import ollama
import json
import os

from sentence_transformers import SentenceTransformer



# 1. Load embedding model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Check saved files
if os.path.exists("rag_index.faiss") and os.path.exists("chunks.json"):

    print("Loading saved index and chunks...")

    index = faiss.read_index("rag_index.faiss")

    with open("chunks.json", "r", encoding="utf-8") as file:
        chunks = json.load(file)

else:

    print("Building new index...")

    # Read knowledge base
    with open("knowledge.txt", "r", encoding="utf-8") as file:
        text = file.read()

    # Split text into chunks
    chunks = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]

    if not chunks:
        raise ValueError("Knowledge base is empty.")

    # Create embeddings
    chunks_embeddings = model.encode(
        chunks,
        convert_to_numpy=True
    )

    chunks_embeddings = np.asarray(
        chunks_embeddings,
        dtype="float32"
    )

    print("Embeddings shape:", chunks_embeddings.shape)

    # Normalize document embeddings
    faiss.normalize_L2(chunks_embeddings)

    # Create FAISS index
    dimension = chunks_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(chunks_embeddings)

    # Save index and chunks
    faiss.write_index(index, "rag_index.faiss")

    with open("chunks.json", "w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)

    print("Index and chunks saved successfully.")

print("Total vectors in FAISS:", index.ntotal)


# 7. Get user question
question = input("\nAsk question: ").strip().lower()


# 8. Create question embedding
question_embedding = model.encode(
    [question],
    convert_to_numpy=True
)

query_vector = np.asarray(
    question_embedding,
    dtype="float32"
)


# 9. Normalize question embedding
faiss.normalize_L2(query_vector)


# 10. Search for Top-3 relevant chunks
similarities, indices = index.search(
    query_vector,
    3
)


print("\nRetrieved information:")

relevant_contexts = []

for idx, score in zip(indices[0], similarities[0]):
    if idx == -1:
        continue

    chunk = chunks[idx]
    relevant_contexts.append(chunk)

    print(f"\nChunk index: {idx}")
    print(f"Cosine similarity: {score:.4f}")
    print(chunk)
    print("-" * 40)


# 11. Generate answer using Ollama
best_score = similarities[0][0]

if best_score < 0.5:
    print("Not enough relevant information.")
else:
    
    context = "\n\n".join(relevant_contexts)

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
         {
                "role": "system",
                "content": (
                    "Answer the user's question using only the provided context. "
                    "If the answer is not available in the context, say so. "
                    "Do not invent facts."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {question}"
                )
            }
        ]
    )


    # 12. Print final answer
    print("\nGenerated answer:")
    print(response["message"]["content"])

