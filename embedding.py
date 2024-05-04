from gensim.models import Word2Vec

EMBEDDING_DIM = 100

def skipgram_embedding(parent_topic, corpus):
    '''
    Train a skip-gram Word2Vec model on the corpus
    Input:
        parent_topic: a Topic object
        corpus: a list of documents
    Output:
        embeddings: a dictionary of term embeddings
    '''
    model = Word2Vec(vector_size=EMBEDDING_DIM, window=5, min_count=1, workers=-1, sg=1)
    model.build_vocab(corpus)
    model.train(corpus, total_examples=model.corpus_count, epochs=30)
    embeddings = {term: model.wv[term] for term in parent_topic.terms}
    return embeddings
