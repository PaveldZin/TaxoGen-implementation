from sklearn.feature_extraction.text import TfidfVectorizer

def tfidf_vectorizer(corpus, terms):
    term_to_index = {term: i for i, term in enumerate(terms)}
    vectorizer = TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x, vocabulary=terms, token_pattern=None)
    tf_idf_matrix = vectorizer.fit_transform(corpus)
    non_zero_elements = [((i, j), tf_idf_matrix[i,j]) for i, j in zip(*tf_idf_matrix.nonzero())]
    return non_zero_elements, term_to_index

