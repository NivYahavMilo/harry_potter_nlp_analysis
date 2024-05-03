import os
import warnings

from langchain_core._api import LangChainDeprecationWarning

warnings.filterwarnings("ignore", category=DeprecationWarning)

RAG_MODEL = 'gpt-3.5-turbo-0125'
RAG_EMBEDDINGS = 'text-embedding-3-large'
OPENAI_KEY = 'sk-proj-Bn1lT27x8oblgbfnVTeDT3BlbkFJV4aphdjC5PgQk1KNUP1D'

os.environ['OPENAI_API_KEY'] = OPENAI_KEY

DATASET = "J. K. Rowling - Harry Potter 1 - Sorcerer's Stone.txt"
