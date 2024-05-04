from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_vectorizer(corpus, terms):
    vectorizer = TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x, vocabulary=terms, token_pattern=None)
    tf_idf_matrix = vectorizer.fit_transform(corpus)
    term_to_index = {term: i for i, term in enumerate(terms)}
    return tf_idf_matrix, term_to_index

