import numpy as np

# numbers = np.array([10, 20, 30, 40])

# print(numbers)
# print(numbers.shape)
# print(numbers[0])
# print(numbers[1])

# a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

# b = a.reshape(3, 4)

# print(b)

import numpy as np

embeddings = np.array([
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
    [0.7, 0.8, 0.9]
], dtype='float32')

print(embeddings.shape)
print(embeddings.dtype)
print(embeddings.sum(axis=0))
print(embeddings[1, 2])