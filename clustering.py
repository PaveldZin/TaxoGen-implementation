import numpy as np

from s_kmeans.spherical_kmeans import SphericalKmeans
from data import Topic
from scores import ScoreCalculator
from copy import deepcopy


def perform_clustering(parent_topic, k, random_state=None, verbose=False):
    '''
    Performs spherical k-means clustering on the terms in a parent topic.
    Input:
        parent_topic: a Topic object
        k: the number of clusters
    Output:
        a list of Topic objects, one for each cluster
    '''

    embeddings_array = np.array([parent_topic.embeddings[term] for term in parent_topic.terms])

    kmeans = SphericalKmeans(n_clusters=k, random_state=random_state)
    if verbose:
        print("performing clustering...")
        
    kmeans.fit(embeddings_array)

    term_cluster_labels = kmeans.labels_

    child_topics = [Topic() for _ in range(k)]
    for label, term in zip(term_cluster_labels, parent_topic.terms):
        child_topics[label].terms.append(term)
        child_topics[label].embeddings[term] = parent_topic.embeddings[term]
        
    return child_topics


def assign_documents_to_clusters(child_topics, corpus, tf_idf_matrix, term_to_index, verbose=False):
    '''
    Assigns each document in the corpus to the cluster with the biggest tf-idf weight of terms.
    Input:
        child_topics: a list of Topic objects
        corpus: list of all documents
        tf_idf_matrix: a sparse matrix of tf-idf weights
        term_to_index: a dictionary mapping terms to indices
    '''
    if verbose:
        print("assigning documents to clusters...")
        
    for i in range(len(corpus)):
        if verbose and i % 1000 == 0:
            print(f'Assigning document {i} out of {len(corpus)}')
        doc_vector = tf_idf_matrix[i].toarray().flatten()
        max_cluster = np.argmax([np.sum([doc_vector[term_to_index[term]] for term in child_topic.terms])
                                 for child_topic in child_topics])
        child_topics[max_cluster].doc_ids.append(i)
    
    return child_topics


def adaptive_clustering(parent_topic, corpus, tf_idf_matrix, term_to_index, k, delta, max_iter, root=False,
                        random_state=None, verbose=False):
    '''
    Performs adaptive clustering on the terms in a parent topic.
    Releases terms that are not representative from child topics back to parent.
    Input:
        parent_topic: a Topic object
        k: the number of clusters
        delta: the threshold for representativeness score
        root: whether the parent topic is the root topic
    Output:
        new_parent_topic: a Topic object
        child_topics: a list of Topic objects
    '''
        
    C_sub = deepcopy(parent_topic)
    
    new_parent_topic = Topic(terms=[], doc_ids=parent_topic.doc_ids,
                             embeddings={}, scores={})
    child_topics = []
    for i in range(max_iter):
        if verbose:
            print("adaptive clustering iteration №", i)
            
        child_topics = perform_clustering(C_sub, k, random_state, verbose)
        
        child_topics = assign_documents_to_clusters(child_topics, corpus, tf_idf_matrix, term_to_index, verbose)
        
        if verbose:
            print("calculating representativeness scores...")
            
        score_calulator = ScoreCalculator(child_topics, corpus)
        
        for topic_index, child_topic in enumerate(child_topics):
            new_child_topic = deepcopy(child_topic)
            for term in child_topic.terms:
                new_child_topic.scores[term] = score_calulator.representativeness(term, topic_index)
                if new_child_topic.scores[term] < delta:
                    # add term to parent topic
                    new_parent_topic.terms.append(term)
                    new_parent_topic.embeddings[term] = parent_topic.embeddings[term]
                    if not root:
                        new_parent_topic.scores[term] = parent_topic.scores[term]
                    # remove term from child topic
                    new_child_topic.terms.remove(term)
                    del new_child_topic.embeddings[term]
                    del new_child_topic.scores[term]
                    # remove term from clustering
                    C_sub.terms.remove(term)
                    del C_sub.embeddings[term]
                    if not root:
                        del C_sub.scores[term]
            child_topics[topic_index] = new_child_topic
            
    return new_parent_topic, child_topics