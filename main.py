from taxonomy import TaxonomyNode
from clustering import adaptive_clustering
from doc_embeddings import compute_doc_embeddings, expand_local_docs
from embedding import skipgram_embedding
from tf_idf_vectorizer import tfidf_vectorizer
from data import Topic

MAX_LEVEL = 2


def recursion(node, corpus, level, tf_idf_matrix, term_to_index, k, n_expand, delta, max_iter, verbose, root=True):
    '''
    Recursively build a taxonomy tree by performing adaptive clustering and local embedding on each node.
    Input:
        node: a Node object
        corpus: a list of documents
        level: the current level in the taxonomy tree
        tf_idf_matrix: a sparse matrix of tf-idf values
        term_to_index: a dictionary mapping terms to their indices in the tf-idf matrix
        k: the number of clusters
        n_expand: the number of documents to add to each child topic by mean direction expansion
        delta: the threshold for representativeness score
        max_iter: the maximum number of iterations for adaptive clustering
        verbose: whether to print progress
    '''
    if level > MAX_LEVEL:
        return
    
    if verbose:
        print(f'Building taxonomy at level {level}, node "{node.name}"')
        
    parent, child_topics = adaptive_clustering(parent_topic=node.data, corpus=corpus, tf_idf_matrix=tf_idf_matrix,
                                               term_to_index=term_to_index, k=k, delta=delta, max_iter=max_iter,
                                               root=root, verbose=verbose)
        
    node.data = parent
    if not root:
        node.name = node.data.top_terms(1)[0]

    
    if level < MAX_LEVEL:
        doc_embeddings = compute_doc_embeddings(corpus, node.data, tf_idf_matrix, term_to_index)
        for i in range(k):
            child_topics[i] = expand_local_docs(child_topics[i], doc_embeddings, n_expand)
            child_topics[i].embeddings = skipgram_embedding(child_topics[i], corpus)

    
    node.children = [TaxonomyNode(data=child_topics[i], name=child_topics[i].top_terms(1)[0]) for i in range(k)]
    
    for i in range(k):
        recursion(node.children[i], corpus, level + 1, tf_idf_matrix, term_to_index, k, n_expand, delta, max_iter,
                  verbose, root=False)

def main(documents, terms, embeddings, k=5, n_expand=100, delta=0.25, max_iter=1, verbose=False):
    '''
    Build a taxonomy tree by performing adaptive clustering and local embedding.
    Input:
        documents: a list of documents (list of lists of terms)
        terms: a list of terms
        k: the number of clusters
        n_expand: the number of documents to add to each child topic by mean direction expansion
        delta: the threshold for representativeness score
        max_iter: the maximum number of iterations for adaptive clustering
        verbose: whether to print progress
    Output:
        root: a TaxonomyNode object representing the root of the taxonomy tree
    '''
    tf_idf_matrix, term_to_index = tfidf_vectorizer(documents, terms)
    root_topic = Topic(terms, len(documents), embeddings)
    root = TaxonomyNode(name='root', data=root_topic)
    recursion(root, documents, 1, tf_idf_matrix, term_to_index, k, n_expand, delta, max_iter, verbose)
    return root