import math

from collections import defaultdict, Counter

class ScoreCalculator:
    '''
    Handle the calculation of popularity, concentration, and representativeness scores of terms in all subtopics.
    '''
    def __init__(self, sub_topics, corpus):
        self.sub_topics = sub_topics
        self.pseudo_documents = self.create_pseudo_documents(corpus)
        self.pseudo_documents_lengths = [len(document) for document in self.pseudo_documents]
        self.avgdl = sum(self.pseudo_documents_lengths) / len(self.pseudo_documents_lengths) # average pseudo-document length
        self.term_freqs = self.create_term_freqs() # list of counters of term frequencies in pseudo-documents
        self.term_doc_count = self.create_term_doc_count() # dictionary of in how many pseudo-documents a term appears
    
       
    def create_pseudo_documents(self, corpus):
        term_set = set()
        for sub_topic in self.sub_topics:
            term_set.update(sub_topic.terms)
        
        pseudo_documents = []
        for sub_topic in self.sub_topics:
            pseudo_document = []
            for document in [corpus[i] for i in sub_topic.doc_ids]:
                trimmed_document = [term for term in document if term in term_set] # so out-of-vocabulary terms won't lower the score
                pseudo_document.extend(trimmed_document)
            pseudo_documents.append(pseudo_document)
        return pseudo_documents
    
    
    def create_term_freqs(self):
        term_freqs = []
        for pseudo_document in self.pseudo_documents:
            term_freq = Counter(pseudo_document)
            term_freqs.append(term_freq)
        return term_freqs
    
    
    def create_term_doc_count(self):
        term_doc_count = defaultdict(int)
        for i in range(len(self.sub_topics)):
            for term in self.term_freqs[i]:
                term_doc_count[term] += 1
        return term_doc_count
    
    
    def pop(self, term, topic_index):
        '''
        Compute the popularity score of a term in a corpus.
        '''
        term_count = self.term_freqs[topic_index][term]
        if term_count == 0:
            return 0.0
        total_tokens = self.pseudo_documents_lengths[topic_index]
        
        return math.log(term_count + 1) / math.log(total_tokens)    
        
    
    def calculate_bm25(self, tf, dl, total_doc_count, doc_present_count, avgdl, k=1.2, b=0.75):
        '''
        Compute BM25 score of a term in a pseudo-document.
        tf: term frequency in the pseudo-document
        dl: length of the pseudo-document
        total_doc_count: total number of pseudo-documents
        doc_present_count: number of pseudo-documents containing the term
        avgdl: average pseudo-document length
        '''
        if tf == 0:
            return 0.0
        
        score = tf * (k + 1) / (tf + k * (1 - b + b * (dl / avgdl)))
        idf = math.log((total_doc_count - doc_present_count + 0.5) / (doc_present_count + 0.5) + 1)
        return score * idf
    
    
    def con(self, term, topic_index):
        '''
        Compute concentration score of a term in a topic.
        '''
        bm25_list = []
        total_doc_count = len(self.sub_topics)
        doc_present_count = self.term_doc_count[term]
        for i in range(len(self.sub_topics)):
            tf = self.term_freqs[i][term]
            dl = self.pseudo_documents_lengths[i]
            bm25_list.append(self.calculate_bm25(tf, dl, total_doc_count, doc_present_count, self.avgdl))
        return math.exp(bm25_list[topic_index]) / (1 + sum([math.exp(score) for score in bm25_list]))
    
    
    def representativeness(self, term, topic_index):
        '''
        Compute the representativeness score of a term in a topic.
        '''
        return math.sqrt(self.pop(term, topic_index) * self.con(term, topic_index))
    
