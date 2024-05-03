import matplotlib.pyplot as plt
import nltk.tokenize
import numpy as np
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from colorama import Fore, Style

import config
import utils


def evaluate_vector_space(vectors):
    """
    Evaluate the generated vector space by computing various metrics.

    Args:
        vectors (np.array): The embedding vectors.

    Returns:
        None
    """
    # Compute norms of vectors
    norms = np.linalg.norm(vectors, axis=1)
    norm_variance = np.var(norms)
    print(Fore.YELLOW + "Variance of vector norms:", norm_variance)

    # Compute cosine similarity matrix
    cosine_similarities = cosine_similarity(vectors)

    # Compute mean cosine similarity
    mean_cosine_similarity = np.mean(cosine_similarities)
    print("Mean cosine similarity:", mean_cosine_similarity)

    # Compute nearest neighbor distances
    nearest_neighbor_distances = np.min(cosine_similarities, axis=1)

    # Compute variance of nearest neighbor distances
    nn_distance_variance = np.var(nearest_neighbor_distances)
    print("Variance of nearest neighbor distances:", nn_distance_variance)

    # Visualize evaluation metrics
    visualize_evaluation_metrics(norms, nearest_neighbor_distances, cosine_similarities)


def visualize_evaluation_metrics(norms, nearest_neighbor_distances, cosine_similarities):
    """
    Visualize the evaluation metrics.

    Args:
        norms (np.array): Vector norms.
        nearest_neighbor_distances (np.array): Nearest neighbor distances.
        cosine_similarities (np.array): Cosine similarities.

    Returns:
        None
    """
    # Plot distribution of vector norms
    plt.figure(figsize=(8, 6))
    sns.histplot(norms, bins=50, kde=True, color='skyblue')
    plt.title('Distribution of Vector Norms')
    plt.xlabel('Norm')
    plt.ylabel('Frequency')
    plt.show()

    # Plot distribution of nearest neighbor distances
    plt.figure(figsize=(8, 6))
    sns.histplot(nearest_neighbor_distances, bins=50, kde=True, color='salmon')
    plt.title('Distribution of Nearest Neighbor Distances')
    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.show()

    # Plot distribution of cosine similarities
    plt.figure(figsize=(8, 6))
    sns.histplot(cosine_similarities.flatten(), bins=50, kde=True, color='green')
    plt.title('Distribution of Cosine Similarities')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.show()


def semantic_retrieval(queries, top_k=5, evaluate_embeddings_space: bool = False):
    """
    Perform semantic retrieval based on input queries.

    Args:
        queries (list): List of queries.
        top_k (int): Number of top passages to retrieve. Defaults to 5.
        evaluate_embeddings_space (bool): Whether to evaluate the embeddings space. Defaults to False.

    Returns:
        None
    """
    corpus = utils.load_dataset(data_file=config.DATASET)

    print("Loading model...")
    # Load miniLM model and tokenizer
    model = SentenceTransformer(config.MINI_LM)

    # Tokenize the text into sentences
    sentences = nltk.tokenize.sent_tokenize(corpus)

    print(f"Creating vector space out of {len(sentences)} sentences...")
    # Encode sentences using the model
    sentence_embeddings = model.encode(sentences, convert_to_numpy=True)

    if evaluate_embeddings_space:
        print("Evaluating vector space...")
        evaluate_vector_space(sentence_embeddings)

    for query in queries:
        # Embed the query
        query_embedding = model.encode(query)

        # Compute cosine similarity between query embedding and sentence embeddings
        similarity_scores = cosine_similarity(query_embedding.reshape(1, -1), sentence_embeddings)[0]

        # Rank sentences based on similarity scores
        ranked_indices = np.argsort(similarity_scores)[::-1]

        # Retrieve top passages
        print(Fore.BLUE + "Query:", query)
        for idx in ranked_indices[:top_k]:
            print(Fore.RED + "Similarity Score:", similarity_scores[idx])
            print(Fore.GREEN + "Passage:", sentences[idx])
        print(Style.RESET_ALL)


if __name__ == '__main__':
    semantic_retrieval(
        queries=[
            "What magical objects are featured in Harry Potter and the Philosopher's Stone?",
            "What is the name of the school attended by wizards in Harry Potter?",
            "Who are Harry Potter's friends in Harry Potter and the Philosopher's Stone?"
        ],
        evaluate_embeddings_space=True,
        top_k=5

    )
