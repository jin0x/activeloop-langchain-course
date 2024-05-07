# https://github.com/peterw/Chat-with-Github-Repo/tree/main

from dotenv import load_dotenv
load_dotenv()
import os

import streamlit as st
from streamlit_chat import message
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

root_dir = "./path/to/cloned/repository" # use any local repository stored on your system
docs = []
file_extensions = []

for dirpath, dirnames, filenames in os.walk(root_dir):

    for file in filenames:
        file_path = os.path.join(dirpath, file)
        if file_extensions and os.path.splitext(file)[1] not in file_extensions:
            continue

    loader = TextLoader(file_path, encoding="utf-8")
    docs.extend(loader.load_and_split())

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
splitted_text = text_splitter.split_documents(docs)

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# Create DeepLake instance and add documents as embeddings
my_activeloop_org_id = os.environ["ACTIVELOOP_ORG_ID"]
my_activeloop_dataset_name = "langchain_course_chat_with_gh"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"

db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)
db.add_documents(splitted_text)

# Create a retriever from the DeepLake instance
retriever = db.as_retriever()

# Set the search parameters for the retriever
retriever.search_kwargs["distance_metric"] = "cos"
retriever.search_kwargs["fetch_k"] = 100
retriever.search_kwargs["k"] = 10

# Create a ChatOpenAI model instance
model = ChatOpenAI()

# Create a RetrievalQA instance from the model and retriever
qa = RetrievalQA.from_llm(model, retriever=retriever)

# Return the result of the query
qa.run("What is the repository's name?")

## STREAMLIT UI ##

# Set the title for the Streamlit app
st.title(f"Chat with GitHub Repository")

# Initialize the session state for placeholder messages.
if "generated" not in st.session_state:
	st.session_state["generated"] = ["i am ready to help you sir"]

if "past" not in st.session_state:
	st.session_state["past"] = ["hello"]

# A field input to receive user queries
user_input = st.text_input("", key="input")

# Search the database and add the responses to state
if user_input:
	output = qa.run(user_input)
	st.session_state.past.append(user_input)
	st.session_state.generated.append(output)

# Create the conversational UI using the previous states
if st.session_state["generated"]:
	for i in range(len(st.session_state["generated"])):
		message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
		message(st.session_state["generated"][i], key=str(i))