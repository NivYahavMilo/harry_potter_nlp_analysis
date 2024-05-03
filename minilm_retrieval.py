import matplotlib.pyplot as plt
import nltk.tokenize
import numpy as np
import seaborn as sns
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import config
import utils


def evaluate_vector_space(vectors):
    # Compute norms of vectors
    norms = np.linalg.norm(vectors, axis=1)
    norm_variance = np.var(norms)
    print("Variance of vector norms:", norm_variance)

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

    # You can add more evaluation metrics as needed

    visualize_evaluation_metrics(norms, nearest_neighbor_distances, cosine_similarities)


def visualize_evaluation_metrics(norms, nearest_neighbor_distances, cosine_similarities):
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


def embed_query(query):
    # Embed the query using the same embedding model used for generating the embedding matrix
    # You need to implement this function based on the specifics of your embedding model
    pass


def semantic_retrieval(queries, top_k=5, evaluate_embeddings_space: bool = False):
    corpus = utils.load_dataset(data_file=config.DATASET)

    # Load miniLM model and tokenizer
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Tokenize the text into sentences
    sentences = nltk.tokenize.sent_tokenize(corpus)

    # Encode sentences using the model
    sentence_embeddings = model.encode(sentences, convert_to_numpy=True)

    if evaluate_embeddings_space:
        evaluate_vector_space(sentence_embeddings)

    for query in queries:
        # Embed the query
        query_embedding = model.encode(query)

        # Compute cosine similarity between query embedding and sentence embeddings
        similarity_scores = cosine_similarity(query_embedding.reshape(1, -1), sentence_embeddings)[0]

        # Rank sentences based on similarity scores
        ranked_indices = np.argsort(similarity_scores)[::-1]

        # Retrieve top passages
        top_passages = [(sentences[idx], similarity_scores[idx]) for idx in ranked_indices[:top_k]]
        print("Query:", query)
        for passage, similarity_score in top_passages:
            print("Similarity Score:", similarity_score)
            print("Passage:", passage)


if __name__ == '__main__':
    semantic_retrieval(
        queries=["What magical objects are featured in Harry Potter and the Philosopher's Stone?",
                 "What is the name of the school attended by wizards in Harry Potter?",
                 "Who are Harry Potter's friends in Harry Potter and the Philosopher's Stone?"

                 ],

    )
