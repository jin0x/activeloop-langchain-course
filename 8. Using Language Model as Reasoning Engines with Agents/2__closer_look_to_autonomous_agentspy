from dotenv import load_dotenv
load_dotenv()
import os

# Importing necessary modules
import faiss
from langchain_openai import OpenAI
from langchain_community.docstore import InMemoryDocstore
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_experimental.autonomous_agents import BabyAGI

# Define the embedding model
embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002")

# Initialize the vectorstore
embedding_size = 1536
index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

# set the goal
goal = "Plan a trip to the Grand Canyon"

# create the babyagi agent
# If max_iterations is None, the agent may go on forever if stuck in loops
baby_agi = BabyAGI.from_llm(
    llm=OpenAI(model="gpt-3.5-turbo-instruct", temperature=0),
    vectorstore=vectorstore,
    verbose=False,
    max_iterations=3
)
response = baby_agi({"objective": goal})