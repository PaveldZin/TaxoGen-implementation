import numpy as np

from s_kmeans.spherical_kmeans import SphericalKmeans
from data import Topic
from scores import ScoreCalculator
from copy import deepcopy
from tqdm import tqdm


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

    kmeans = SphericalKmeans(n_clusters=k, tol=1e-9, max_iter=1000, random_state=random_state)
    if verbose:
        print("\tperforming clustering...")
        
    kmeans.fit(embeddings_array)

    child_topics = [Topic() for _ in range(k)]
    for label, term in zip(kmeans.labels_, parent_topic.terms):
        child_topics[label].terms.append(term)
        child_topics[label].embeddings[term] = parent_topic.embeddings[term]
        
    return child_topics


def assign_documents_to_clusters(child_topics, corpus, non_zero_elements, term_to_index, verbose=False):
    '''
    Assigns each document in the corpus to the cluster with the biggest tf-idf weight of terms.
    Input:
        child_topics: a list of Topic objects
        corpus: list of all documents
        non_zero_elements: non zero elements of a sparse matrix of tf-idf weights
        term_to_index: a dictionary mapping terms to indices
    '''
        
    # for i in tqdm(range(len(corpus)), disable=not verbose, desc='\tassigning documents to clusters'):
        # doc_vector = tf_idf_matrix[i].toarray().flatten()
        # max_cluster = np.argmax([np.sum([doc_vector[term_to_index[term]] for term in child_topic.terms])
        #                          for child_topic in child_topics])
        # child_topics[max_cluster].doc_ids.append(i)
    term_index_to_topic_index = {term_to_index[term]: topic_index for topic_index, child_topic in enumerate(child_topics) for term in child_topic.terms}
    #non_zero_elements = [((i, j), tf_idf_matrix[i,j]) for i, j in zip(*tf_idf_matrix.nonzero())]
    topic_scores = np.zeros((len(corpus), len(child_topics)))
    for (i, j), value in non_zero_elements:
        if j in term_index_to_topic_index:
            topic_index = term_index_to_topic_index[j]
            topic_scores[i, topic_index] += value
    if verbose:
        print("\tassigning documents to clusters...")
    for i in range(len(corpus)):
        if np.any(topic_scores[i]):
            max_cluster = np.argmax(topic_scores[i])
            child_topics[max_cluster].doc_ids.append(i)
    return child_topics


def adaptive_clustering(parent_topic, corpus, non_zero_elements, term_to_index, k, delta, max_iter, root=False,
                        random_state=None, verbose=False):
    '''
    Performs adaptive clustering on the terms in a parent topic.
    Releases terms that are not representative from child topics back to parent.
    Input:
        parent_topic: a Topic object
        corpus: list of all documents
        non_zero_elements: non zero elements of a sparse matrix of tf-idf weights
        term_to_index: a dictionary mapping terms to indices        
        k: the number of clusters
        delta: the threshold for representativeness score
        max_iter: the maximum number of iterations
        root: whether the parent topic is the root topic
    Output:
        new_parent_topic: a Topic object
        child_topics: a list of Topic objects
    '''
        
    C_sub = deepcopy(parent_topic)
    new_parent_topic = Topic(doc_ids=parent_topic.doc_ids)
    child_topics = []
    for i in range(max_iter):
        if verbose:
            print(f"\tadaclust iteration {i + 1}/{max_iter}")
        
        child_topics = perform_clustering(C_sub, k, random_state, verbose)
        
        child_topics = assign_documents_to_clusters(child_topics, corpus, non_zero_elements, term_to_index, verbose)
        
        if verbose:
            print("\tcalculating representativeness scores...")
            
        score_calulator = ScoreCalculator(child_topics, corpus)
        
        for topic_index, child_topic in enumerate(child_topics):
            new_child_topic = deepcopy(child_topic)
            for term in child_topic.terms:
                new_child_topic.scores[term] = score_calulator.representativeness(term, topic_index)
                if new_child_topic.scores[term] < delta:
                    # add term back to parent topic
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