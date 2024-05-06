from dotenv import load_dotenv
load_dotenv()

import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import DeepLake
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# create our documents
texts = [
    "Napoleon Bonaparte was born in 15 August 1769",
    "Louis XIV was born in 5 September 1638",
    "Lady Gaga was born in 28 March 1986",
    "Michael Jeffrey Jordan was born in 17 February 1963"
]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.create_documents(texts)

# initialize embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# create Deep Lake dataset
my_activeloop_org_id = os.environ["ACTIVELOOP_ORG_ID"]
my_activeloop_dataset_name = "langchain_course_embeddings"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path, embedding=embeddings, overwrite=True)

# add documents to our Deep Lake dataset
db.add_documents(docs)

# create retriever from db
retriever = db.as_retriever()

# istantiate the llm wrapper
model = ChatOpenAI(model='gpt-3.5-turbo')

# create the question-answering chain
qa_chain = RetrievalQA.from_llm(model, retriever=retriever)

# ask a question to the chain
response = qa_chain.invoke("When was Michael Jordan born?")

print(response)
