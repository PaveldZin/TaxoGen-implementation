# from scores import pop


class Topic:
    def __init__(self, terms=None, doc_ids=None, embeddings=None, scores=None):
        '''
        A Topic object is the node of the taxonomy tree.
        terms: a list of terms in the topic
        doc_ids: a list of document indexes in the topic
        embeddings: a dictionary of term embeddings
        scores: a dictionary of representativeness scores for each term
        '''
        self.terms = terms if terms else []
        self.doc_ids = doc_ids if doc_ids else []
        self.embeddings = embeddings if embeddings else {}
        self.scores = scores if scores else {}

    def top_terms(self, n=5):
        if self.scores:
            return sorted(self.scores, key=self.scores.get, reverse=True)[:n]
        else:
            raise ValueError('Scores have not been calculated yet or it is root topic')