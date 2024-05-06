from dotenv import load_dotenv
load_dotenv()

import os
import re
import requests
from bs4 import BeautifulSoup
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader


# Perform the indexing process and upload embeddings to Deep Lake
my_activeloop_org_id = os.environ["ACTIVELOOP_ORG_ID"]
my_activeloop_dataset_name = "langchain_course_jarvis_assistant"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"

# Load the dataset
db = DeepLake(dataset_path=dataset_path, embedding=embeddings)


embeddings =  OpenAIEmbeddings(model="text-embedding-ada-002")
