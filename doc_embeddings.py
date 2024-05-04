import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity


def compute_doc_embeddings(corpus, parent_topic, tf_idf_matrix, term_to_index):
    """
    Compute document embeddings as tf-idf weighted average of term embeddings
    """
    doc_embeddings = []
    for i, doc in enumerate(corpus):
        doc_embedding = np.zeros(len(parent_topic.embeddings[parent_topic.terms[0]]))
        term_count = 0
        for term in parent_topic.terms:
            if term in doc:
                term_embedding = parent_topic.embeddings[term]
                term_index = term_to_index[term]
                term_weight = tf_idf_matrix[i].toarray().flatten()[term_index]
                doc_embedding += term_weight * term_embedding
                term_count += 1
        if term_count > 0:
            doc_embedding /= term_count
        doc_embeddings.append(doc_embedding)

    return doc_embeddings


def get_top_closest_documents(child_topic, doc_embeddings, n):
    """
    Find the n documents closest to the mean direction of the topic embeddings
    """
    query_vec = np.mean(normalize([child_topic.embeddings[term] for term in child_topic.terms], axis=1), axis=0)
    
    distances = cosine_similarity([query_vec], doc_embeddings)[0]
    best_indexes = np.argsort(distances)[::-1][:n]
    return best_indexes


def expand_local_docs(child_topic, doc_embeddings, n):
    """
    Expand the documents in the topic by adding the n closest documents to the mean direction of the topic embeddings
    Input:
        child_topic: a Topic object
        doc_embeddings: a list of document embeddings
        n: the number of documents to add
    """
    best_indexes = get_top_closest_documents(child_topic, doc_embeddings, n)
    child_topic.doc_ids.extend(best_indexes)
    child_topic.doc_ids = list(set(child_topic.doc_ids))
    return child_topic
