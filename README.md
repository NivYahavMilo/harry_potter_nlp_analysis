# harry_potter_nlp_analysis

## Requirements
``
pip install -r requirements.txt
``

### Lexical Retrieval Script
This script implements a retrieval engine based on TF-IDF similarity for retrieving passages from a given dataset.

run with the following command

``python lexical_retrieval.py --top_k 5
``

### MiniLM Retrieval Script
This script performs semantic retrieval using the MiniLM model to encode input queries and retrieve top passages based on cosine similarity.

run with the following command

``
python minilm_retrieval.py --top_k 5 --evaluate_embeddings_space
``

### Harry Potter Passage Retrieval System!

This script performs interactive passages retrieval using the Retrieval-Augmented Generation (RAG) model.

run with the following command:

``python retrieval_augmented_generation.py
``