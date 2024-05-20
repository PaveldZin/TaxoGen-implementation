from gensim.models import Word2Vec, Doc2Vec, FastText
from gensim.utils import effective_n_jobs
from gensim.models.callbacks import CallbackAny2Vec

EMBEDDING_DIM = 100

class callback(CallbackAny2Vec):
    '''Callback to print loss after each epoch.'''

    def __init__(self):
        self.epoch = 0
        self.loss_to_be_subed = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        loss_now = loss - self.loss_to_be_subed
        self.loss_to_be_subed = loss
        print('Loss after epoch {}: {}'.format(self.epoch, loss_now))
        self.epoch += 1
        
def skipgram_embedding(terms, corpus, n_jobs=1):
    '''
    Train a skip-gram Word2Vec model on the corpus
    Input:
        terms: terms to embed
        corpus: a list of documents
    Output:
        embeddings: a dictionary of term embeddings
    '''
        
    model = Word2Vec(vector_size=EMBEDDING_DIM, window=5, min_count=1, negative=5,
                     workers=effective_n_jobs(n_jobs), sg=1)
    model.build_vocab(corpus)
    model.train(corpus, total_examples=model.corpus_count, epochs=5)
    embeddings = {term: model.wv[term] for term in terms}
    return embeddings

def doc2vec_embedding(corpus, n_jobs=1):
    '''
    Train a Doc2Vec model on the corpus
    '''
    model = Doc2Vec(vector_size=EMBEDDING_DIM, window=5, min_count=10, workers=effective_n_jobs(n_jobs))
    model.build_vocab(corpus)
    model.train(corpus, total_examples=model.corpus_count, epochs=10)
    return model

def fasttext_embedding(terms, corpus, n_jobs=1):
    '''
    Train a fastText model on the corpus
    Input:
        terms: terms to embed
        corpus: a list of documents
    Output:
        embeddings: a dictionary of term embeddings
    '''
        
    model = FastText(vector_size=EMBEDDING_DIM, window=5, min_count=1, negative=5,
                     workers=effective_n_jobs(n_jobs), sg=1)
    model.build_vocab(corpus)
    model.train(corpus, total_examples=model.corpus_count, epochs=5)
    embeddings = {term: model.wv[term] for term in terms}
    return embeddings