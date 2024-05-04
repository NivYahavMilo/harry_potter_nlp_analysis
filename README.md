# harry_potter_nlp_analysis

## Requirements

install if you intend to run all functionalities.

``
pip install -r requirements.txt
``

### Lexical Retrieval Script

#### requirements:
``
nltk==3.8.1
pandas==2.2.2
colorama==0.4.6
scikit-learn==1.4.2
``

This script implements a retrieval engine based on TF-IDF similarity for retrieving passages from a given dataset.

run with the following command


``
python3 lexical_retrieval.py --top_k 5
``

### MiniLM Retrieval Script
This script performs semantic retrieval using the MiniLM model to encode input queries and retrieve top passages based on cosine similarity.

#### requirements:
``
matplotlib==3.8.4
seaborn==0.13.2
sentence-transformers==2.7.0
``

eval flag: 
Flag to evaluate the embeddings space. Include this flag to visualize evaluation metrics.

run with the following command

``
python3 minilm_retrieval.py --top_k 5 --eval
``


### Harry Potter Passage Retrieval System!

This script performs interactive passages retrieval using the Retrieval-Augmented Generation (RAG) model.

#### requirements:
````
langchain==0.1.17
llama-index==0.10.34
llama-index-llms-langchain==0.1.3
llama-index-embeddings-langchain==0.1.2
````

run with the following command:

``python3 retrieval_augmented_generation.py
``