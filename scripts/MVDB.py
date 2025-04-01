from sentence_transformers import SentenceTransformer
from PIL import Image
import numpy as np
import json
import torch
import timeit
import faiss
import sqlite3
import time

class Mode:
    def __init__(self, model, dim, metric, datatype, data, ifile, efile):
        self.model = model
        self.dim = dim
        self.metric = metric
        self.datatype = datatype
        self.data = data
        self.embeddings = None
        self.efile = efile
        self.index = None
        self.ifile = ifile

    def embed_query(self, data):
        if self.datatype=="TEXT":
            query = np.array(self.model.encode(data)).reshape(1, -1)
        elif self.datatype=="IMAGE":
            query = np.array(self.model.encode(Image.open(data))).reshape(1, -1)
        # modify later
        faiss.normalize_L2(query)
        return query
    
    def search(self, query, k):
        D, I = self.index.search(query, k=k)
        D = D.reshape(D.shape[1])
        I = I.reshape(I.shape[1])
        similarities = D[np.argsort(I)]
        return similarities

    def create_embeddings(self):
        if self.datatype=="TEXT":
            embeddings = self.model.encode(self.data, show_progress_bar=False)
        elif self.datatype=="IMAGE":
            bs = 1000
            embeddings = np.zeros((len(self.data), self.dim))
            for i in range(0, len(self.data), bs):
                batch = self.data[i:i+bs]
                batch_embeddings = self.model.encode([Image.open(path) for path in batch], show_progress_bar=False)
                embeddings[i:i+bs,:] = batch_embeddings
        self.embeddings = embeddings.astype(np.float32)

    def create_index(self, use_precomputed_embeddings=False):
        if(use_precomputed_embeddings):
            data = np.load(self.efile)
            self.embeddings = data['arr_0']
            data.close()
            print('loaded embeddings')
        else:
            self.create_embeddings()
            print('created embeddings')

        # modify later for different metrics
        faiss.normalize_L2(self.embeddings)
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(self.embeddings)

    def save(self):
        np.savez_compressed(self.efile, arr_0=self.embeddings)
        faiss.write_index(self.index, self.ifile)

    def load(self):
        data = np.load(self.efile)
        self.embeddings = data['arr_0']
        data.close()
        self.index = faiss.read_index(self.ifile)

class MVDB:
    def __init__(self, modes):
        # Settings
        self.modes = modes

    def create_indexes(self, use_precomputed_embeddings=False):
        total_dim = 0
        for mode in self.modes:
            mode.create_index(use_precomputed_embeddings)
            total_dim+=mode.dim

        print('created indexes')

    # Saving/Loading
    def save_indexes(self):
        for mode in self.modes:
            mode.save()

    def load_indexes(self):
        for mode in self.modes:
            mode.load()

    # Querying
    def get_query(self, data):
        query = []
        for i, mode in enumerate(self.modes):
            query.append(mode.embed_query(data[i]))
        return query

    def naive_search(self, query, k, aggregation_fn):
        similarities = []
        for i, mode in enumerate(self.modes):
            similarities.append(mode.search(query[i], k=mode.index.ntotal))

        aggregated_similarities = aggregation_fn(similarities)

        # Get top k
        knn = np.argpartition(aggregated_similarities,-k)[-k:]
        indexes = knn[np.argsort(aggregated_similarities[knn])][::-1]
        scores = aggregated_similarities[indexes]

        return indexes, scores

    # Aggregation
    def average(similarities):
        aggregated = similarities[0]
        for i in range(1, len(similarities)):
            aggregated += similarities[i]
        aggregated = aggregated / len(similarities)
        return aggregated