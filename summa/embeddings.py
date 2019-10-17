from flair.embeddings import WordEmbeddings, FlairEmbeddings, DocumentPoolEmbeddings, Sentence
import torch.nn as nn


flair_embedding_forward = FlairEmbeddings('news-forward')
flair_embedding_backward = FlairEmbeddings('news-backward')
embedding = DocumentPoolEmbeddings([flair_embedding_forward, flair_embedding_backward])

cos = nn.CosineSimilarity(dim=0)

def _get_embedding(text):
    sentence = Sentence(text)
    embedding.embed(sentence)
    vector = sentence.get_embedding()
    return vector

def _get_cosine_similarity(vec_1, vec_2):
    return round(cos(vec_1, vec_2).item(), 3) 

def get_embedding_similarity(sentence_1, sentence_2):
    vec_1 = _get_embedding(sentence_1)
    vec_2 = _get_embedding(sentence_2)
    return _get_cosine_similarity(vec_1, vec_2)