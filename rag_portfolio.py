
from sentence_transformers import SentenceTransformer, util
import ollama

# Read knowledge base
with open('knowledge.txt', 'r') as file:
    text = file.read()

# Split text into chunks
chunks = [
    paragraph.strip()
    for paragraph in text.split('\n\n')
    if paragraph.strip()
]

# Load embedding model
model = SentenceTransformer(
    'sentence-transformers/all-MiniLM-L6-v2'
)

# Create embeddings for chunks
chunks_embeddings = model.encode(chunks)

# Get user question
question = input('Ask question: ')

# Create question embedding
question_embedding = model.encode(question)

# Calculate similarities
similarities = util.cos_sim(
    question_embedding,
    chunks_embeddings
)[0]

# Find Top-3 chunks
similarities_list = []

for index, score in enumerate(similarities):
    similarities_list.append((index, score.item()))

similarities_list.sort(
    key=lambda x: x[1],
    reverse=True
)

top_k = similarities_list[:3]

# Retrieve relevant contexts
relevant_contexts = []

for index, score in top_k:
    relevant_contexts.append(chunks[index])

print('\nRetrieved information:')

for context in relevant_contexts:
    print(context)

# Generate answer using Ollama
response = ollama.chat(
    model='llama3.2:3b',
    messages=[
        {
            'role': 'system',
            'content': (
                'Answer the user question using only the provided context. '
                'If the context does not contain the answer, say that the '
                'information is not available in the provided context. '
                'Do not invent facts.'
            )
        },
        {
            'role': 'user',
            'content': (
                f"Context:\n{'\n\n'.join(relevant_contexts)}\n\n"
                f"Question: {question}"
            )
        }
    ]
)

print('\nGenerated answer:')
print(response['message']['content'])
