import os
from typing import Optional

from colorama import Fore, Style
from langchain.chains.llm import LLMChain
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from llama_index.core import ServiceContext, GPTVectorStoreIndex, Document, StorageContext, load_index_from_storage

import config as retrieval_config
from utils import load_dataset


class RAGPassagesExtractionPipeline:
    """
    A pipeline for retrieving passages using the RAG (Retrieval-Augmented Generation) model.
    """

    def __init__(self, llm: Optional[str] = None, embedding_model: Optional[str] = None):
        """
        Initializes the pipeline.

        Args:
            llm (Optional[str]): Name of the LLM model. Defaults to None.
            embedding_model (Optional[str]): Name of the embedding model. Defaults to None.
        """
        self._embedding_model = OpenAIEmbeddings(model=embedding_model or retrieval_config.RAG_EMBEDDINGS)
        self._llm = ChatOpenAI(model_name=llm or retrieval_config.RAG_MODEL, temperature=0.1, streaming=True,
                               max_tokens=50)
        self._service_context = ServiceContext.from_defaults(llm=self._llm, chunk_size=512,
                                                             embed_model=self._embedding_model)
        self._storage_exists_validation()

    def _storage_exists_validation(self):
        """
        Validates the existence of the storage directory and its contents.
        If not found or incomplete, creates a new vector store index.
        """
        storage_path = os.path.join(os.path.abspath(os.path.curdir), 'storage')
        if not os.path.exists(storage_path):
            print(Fore.RED + "Storage directory does not exist." + Style.RESET_ALL)
            self._create_vector_db_storage()
        elif not all(file in os.listdir(storage_path) for file in ['default__vector_store.json', 'docstore.json',
                                                                   'graph_store.json', 'image__vector_store.json',
                                                                   'index_store.json']):
            print(Fore.RED + "Storage directory exists but is incomplete. Creating a new one..." + Style.RESET_ALL)
            self._create_vector_db_storage()

    def _create_vector_db_storage(self):
        """
        Creates a new vector store index and saves it to disk.
        """
        print(Fore.GREEN + "Creating vector store index as retrieval engine..." + Style.RESET_ALL)
        corpus = load_dataset(data_file=retrieval_config.DATASET)
        document_obj = Document(text=corpus)
        index = GPTVectorStoreIndex.from_documents(
            [document_obj],
            service_context=self._service_context,
            show_progress=True
        )
        index.set_index_id("vector_index")
        index.storage_context.persist("./storage")

    def _prepare_vector_index_for_retrieving(self):
        """
        Prepares the vector index for retrieving passages.
        """
        print("Preparing vector index store...")
        storage_context = StorageContext.from_defaults(persist_dir="storage")
        vector_store_index = load_index_from_storage(
            storage_context=storage_context, index_id="vector_index", service_context=self._service_context
        )
        retriever = vector_store_index.as_retriever(similarity_top_k=5)
        return retriever

    def interactive_passages_retrieval(self, print_context: bool = False):
        """
        Performs interactive passages retrieval based on user input.
        """
        retriever = self._prepare_vector_index_for_retrieving()
        system_prompt = "Given a query received from the user, use the context from the first Harry Potter book " \
                        "to generate the most relevant passages. Keep the answer concise, up to 50 tokens."
        messages = [SystemMessage(content=system_prompt), HumanMessagePromptTemplate.from_template("{input}")]
        prompt = ChatPromptTemplate(messages=messages)
        llm_chain = LLMChain(prompt=prompt, llm=self._llm, verbose=False)
        # Interactive interface
        print(Fore.BLUE + "Welcome to the Harry Potter Passage Retrieval System!\n" + Style.RESET_ALL)
        while True:
            query = input("Enter your search query (type 'exit' to quit): ")
            if not query.strip():
                print(Fore.YELLOW + "Please provide a non-empty search query." + Style.RESET_ALL)
                continue
            if query.lower() == 'exit':
                break
            else:
                print(Fore.YELLOW + f"Retrieving 5 similar passages for query: '{query}'" + Style.RESET_ALL)
                retrieved_passages = retriever.retrieve(str_or_query_bundle=query)
                for passage_node in retrieved_passages:
                    response = llm_chain.run(passage_node.text)
                    print(Fore.GREEN + "Response:\n", response + Style.RESET_ALL)
                    if print_context:
                        print("Book Context:\n", passage_node.text)

                    print(Fore.RED + f"\nPassage Similarity: {passage_node.score}" + Style.RESET_ALL)


if __name__ == '__main__':
    retrieval_pipeline = RAGPassagesExtractionPipeline()
    retrieval_pipeline.interactive_passages_retrieval()
