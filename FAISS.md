## FAISS (Facebook AI Similarity Search):
Vector DB -> Similarity Search and clustering of high dimensional dense vectors.

**Indexing** -> Organizing vectors that allows fast similarity search
i.e) Finding the top K nearest neighbors to a query using Euclidian (L2) or inner product (Normalized vectors for cosine similarity, no effect of magnitude)

### Types of Indexes:
- `IndexFlatL2` / `IndexFlatIP` -> Brute force (exact search), L2 or Inner product
- `IndexIVFFlat` -> Inverted file(ANN) + flat vectors (fast, approximate)
- `IndexIVFPQ` -> IVF + Product Quantization (very memory-efficient)
- `IndexHNSW` -> Hierarchical Navigable Small World graph (ANN, fast, approximate) [More on ANN notes]
- `IndexPQ` -> Product quantization (no IVF), Compresses vectors for memory

### Usage

1. #### IndexFlatL2 / IndexFlatIP
```
d = 512 #vector dimension
index = faiss.IndexFlatL2(d)
index.add(vector)
```
2. #### IndexIVFFlat
FAISS IVF is a ANN method. IVF index must be trained before adding the vectors. In learning phase it learns to cluster vector space `nlist` clusters.
IVF is a clustering based ANN method (partition based).

***Why Training is Needed?*** 
-   Training partitions the vector space using **k-means** clustering.
-   Each added vector is assigned to its nearest centroid (inverted list).
-   At query time, only a few relevant partitions (`nprobe`) are searched, making it fast.

```
# 1. Create a quantizer (base index, typically flat)
quantizer = faiss.IndexFlatL2(d)

# 2. Create IVF index
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

# 3. Train with a subset of data
index.train(xb)    # training learns the centroids

# 4. Add vectors to index
index.add(xb)   # vectors will be assigned to the learned clusters

# 5. Search
index.nprobe = 10   # number of partitions to search (higher = more accurate, slower)
D, I = index.search(xq, k=5)

```

***Tips***

-   `nlist` should be around √(number of vectors), e.g., for 1M vectors, try 1K–10K.
-   Training can be done on a **subset** of your data (e.g., 10k–100k vectors).
-   You can combine IVF with **PQ** (`IndexIVFPQ`) or **Scalar Quantization** (`IndexIVFScalarQuantizer`) for memory efficiency.

***FAISS can combine IVF with:***
-    **Flat** (exact search within each cluster) → `IndexIVFFlat`
-    **PQ (Product Quantization)** → `IndexIVFPQ`
-    **HNSW** as the quantizer → `IndexIVFFlat + HNSW`

3.  #### IndexPQ (Full PQ over all vectors (slow search, but compresses memory)
Instead of storing full-precision vectors (32 bit / 4 byte) float **PQ splits each vector into sub-vectors**, quantizes each sub-vector independently using **k-means**, and stores **only the codebook indices** (compact representation).

https://chatgpt.com/c/6864c730-1548-8005-815b-d02a2ce688ee
https://chatgpt.com/c/6863c723-f188-8005-b401-2549b3cc0a22