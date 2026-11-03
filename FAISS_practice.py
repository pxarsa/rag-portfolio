import faiss
import numpy as np


# vectors = np.array([
#     [1.0, 2.0],
#     [2.0, 3.0],
#     [8.0, 9.0]
# ], dtype='float32')

# index = faiss.IndexFlatL2(2)

# index.add(vectors)

# query = np.array([[8.0, 8.0]], dtype="float32")

# distances, indices = index.search(query, 2)

# print(indices)
# print(distances)
# indices = [[2, 0, 4]]
# print(indices[0][1])