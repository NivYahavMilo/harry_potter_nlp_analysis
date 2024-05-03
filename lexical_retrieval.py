import os

import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity




def create_retrieval_engine():
    corpus = load_dataset(data_file="J. K. Rowling - Harry Potter 1 - Sorcerer's Stone.txt")
    sentences = nltk.tokenize.sent_tokenize(corpus)

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    sentences_df = pd.DataFrame({"sentences": sentences})
    sentences_df.sentences = sentences_df.sentences.str.lower()

    # Fit and transform the corpus to TF-IDF matrix
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences_df.sentences)

    # Define example search queries
    example_queries = ["Harry Potter is Lord Voldemort Nemesis", "Hogwarts Train", "Spells and potions"]

    top_n = 5
    for query in example_queries:
        query = query.lower()
        # Transform the query to TF-IDF vector
        query_tfidf = tfidf_vectorizer.transform([query])

        # Calculate cosine similarity between query and each document
        cosine_similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()

        indices_and_similarities = [(idx, val) for idx, val in enumerate(cosine_similarities)]
        # Sort and keep the top N entries
        indices_and_similarities.sort(key=lambda item: item[1], reverse=True)
        if indices_and_similarities:  # check if list is not empty
            indices, similarities = zip(*indices_and_similarities[:top_n])

        # Get the indices of passages with top TF-IDF scores
        top_passages_indices = cosine_similarities.argsort()[-5:][::-1]

        print(f"Top 5 passages for query '{query}' :")
        for idx, sentence_pos in enumerate(top_passages_indices, 1):
            print(f"Passage {idx}:", sentences[sentence_pos], '\n', "simi")



if __name__ == '__main__':
    # retrieve_documents_tfidf()
    create_retrieval_engine()
