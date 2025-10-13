from openai import OpenAI
import numpy as np

client = OpenAI()

words = ["mathematics", "math", "algebra", "geometry", "calculus",
         "statistics", "physics", "dog", "ice cream", "computer science"]

# Get embeddings
resp = client.embeddings.create(
    model="text-embedding-ada-002",
    input=words
)
embs = np.array([e.embedding for e in resp.data])

# Cosine similarity
def cosine(a,b): return np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
math_emb = embs[0]
sims = [cosine(math_emb, e) for e in embs]

# Rank
friends = sorted(zip(words, sims), key=lambda x: -x[1])
print(friends)
