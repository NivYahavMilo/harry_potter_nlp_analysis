import evaluate

from lexical_retrieval import lexical_retrieving
from minilm_retrieval import semantic_retrieval


def _get_predictions():
    query_list = [
        "What magical objects are featured in the book?",
        "What are the rules in the Quidditch game?",
        "Who is Lord Voldemort?"
    ]

    predicted_results = lexical_retrieving(queries=query_list, top_k=5)
    true_results = semantic_retrieval(queries=query_list, top_k=5, evaluate_embeddings_space=False)

    return true_results, predicted_results


def _calculate_p_at_k_metric(predicted_passages, true_passages, top_k):
    # Calculate Precision@k and Recall@k
    precision_at_k = []
    for true, pred in zip(true_passages, predicted_passages):
        true = true.lower()
        pred = pred.lower()
        num_relevant_pred = len(set(true) & set(pred))
        precision_at_k.append(num_relevant_pred / top_k)

    return {"p@k": precision_at_k}

def _calculate_rouge_and_blue_metrics(predicted_passages, true_passages):
    # Load the metrics
    rouge = evaluate.load('rouge')
    bleu = evaluate.load('bleu')

    # Compute ROUGE scores
    rouge_results = rouge.compute(predictions=predicted_passages, references=true_passages)

    # Compute BLEU scores
    bleu_results = bleu.compute(predictions=predicted_passages, references=true_passages)

    evaluation_results = {"ROUGE": rouge_results, "BLEU": bleu_results}
    return evaluation_results


def evaluate_tf_idf_results():
    """
    Evaluate TF-IDF results against ground truth using Precision@k, Recall@k, MAP, and NDCG metrics.

    Args:
    - tfidf_passages (list of lists): TF-IDF retrieved passages for each query.
    - ground_truth_passages (list of lists): Ground truth passages for each query.
    - k (int): Number of top passages to consider for Precision@k and Recall@k.

    Returns:
    - precision_at_k (float): Precision@k score.
    - recall_at_k (float): Recall@k score.
    - map_score (float): Mean Average Precision (MAP) score.
    - ndcg_score (float): Normalized Discounted Cumulative Gain (NDCG) score.
    """
    query_list = [
        "What magical objects are featured in the book?",
        "What are the rules in the Quidditch game?",
        "Who is Lord Voldemort?"
    ]

    predicted_results = lexical_retrieving(queries=query_list, top_k=5)
    true_results = semantic_retrieval(queries=query_list, top_k=5, evaluate_embeddings_space=False)
    for query in query_list:
        true_passages = true_results[query]
        predicted_passages = predicted_results[query]

        p_at_k_results = _calculate_p_at_k_metric(predicted_passages=predicted_passages, true_passages=true_passages,
                                                  top_k=5)
        rouge_and_blue_results = _calculate_rouge_and_blue_metrics(predicted_passages=predicted_passages,
                                                               true_passages=true_passages)

        evaluation_results = p_at_k_results | rouge_and_blue_results
        print("Evaluation Results:\n")
        print(f"Query: {query}\n")
        print("Results:\n", evaluation_results)


if __name__ == '__main__':
    evaluate_tf_idf_results()
