import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from colorama import Fore, Style

import config
import utils


def create_retrieval_engine(top_k: int):
    """
    Create a retrieval engine based on TF-IDF similarity.

    Args:
        top_k (int): Number of top passages to retrieve.

    Returns:
        None
    """
    corpus = utils.load_dataset(data_file=config.DATASET)
    sentences = nltk.tokenize.sent_tokenize(corpus)

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    sentences_df = pd.DataFrame({"sentences": sentences})
    sentences_df.sentences = sentences_df.sentences.str.lower()

    # Fit and transform the corpus to TF-IDF matrix
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences_df.sentences)

    # Define example search queries
    queries = [
        "What magical objects are featured in Harry Potter and the Philosopher's Stone?",
        "What is the name of the school attended by wizards in Harry Potter?",
        "Who are Harry Potter's friends in Harry Potter and the Philosopher's Stone?"
    ]

    for query in queries:
        query = query.lower()
        # Transform the query to TF-IDF vector
        query_tfidf = tfidf_vectorizer.transform([query])

        # Calculate cosine similarity between query and each document
        cosine_similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()

        # Sort indices based on cosine similarity
        indices_and_similarities = [(idx, val) for idx, val in enumerate(cosine_similarities)]
        indices_and_similarities.sort(key=lambda item: item[1], reverse=True)
        if indices_and_similarities:  # check if list is not empty
            top_passages_indices, _ = zip(*indices_and_similarities[:top_k])

            # Print top passages for the query
            print(Fore.BLUE + f"Top {top_k} passages for query '{query}':" + Style.RESET_ALL)
            for idx, sentence_pos in enumerate(top_passages_indices, 1):
                print(Fore.RED + f"Passage {idx}:" + Style.RESET_ALL, sentences[sentence_pos])


if __name__ == '__main__':
    create_retrieval_engine(top_k=5)
