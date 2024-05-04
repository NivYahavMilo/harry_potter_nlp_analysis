import argparse

import nltk
import pandas as pd
from colorama import Fore, Style
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config
import utils


def lexical_retrieving(queries: list[str], top_k: int):
    """
    Create a retrieval engine based on TF-IDF similarity.

    Args:
        queries (list): queries for passages retrieval.
        top_k (int): Number of top passages to retrieve.

    Returns:
        None
    """
    corpus = utils.load_dataset(data_file=config.DATASET)
    sentences = nltk.tokenize.sent_tokenize(corpus.lower())

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    sentences_df = pd.DataFrame({"sentences": sentences})
    sentences_df.sentences = sentences_df.sentences.str.lower()

    # Fit and transform the corpus to TF-IDF matrix
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences_df.sentences)

    passages_retrieved = {}
    for query in queries:
        query_cased = query.lower()
        # Transform the query to TF-IDF vector
        query_tfidf = tfidf_vectorizer.transform([query_cased])

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
                passage_candidate = sentences[sentence_pos]
                print(Fore.RED + f"Passage {idx}:" + Style.RESET_ALL, passage_candidate)

                # Storing results per query
                passages_retrieved.setdefault(query, []).append(passage_candidate)

    return passages_retrieved


if __name__ == '__main__':

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Retrieve top passages using TF-IDF similarity.')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top passages to retrieve (default: 5)')
    args = parser.parse_args()

    # Define example search queries
    query_list = [
        "What magical objects are featured in the book?",
        "what are the rules in the quidditch game?",
        "Who are Harry Potter's friends in Harry Potter and the Philosopher's Stone?"
    ]
    lexical_retrieving(queries=query_list, top_k=args.top_k)
