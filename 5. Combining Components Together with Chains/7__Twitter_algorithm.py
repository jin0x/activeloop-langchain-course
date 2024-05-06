from dotenv import load_dotenv
load_dotenv()

import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

embeddings = OpenAIEmbeddings()

# Load all files inside the repository.
root_dir = './the-algorithm'
docs = []
for dirpath, dirnames, filenames in os.walk(root_dir):
    for file in filenames:
        try:
            loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
            docs.extend(loader.load_and_split())
        except Exception as e:
            pass

# Divide the loaded files into chunks:
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(docs)

# Check the content of the texts variable
print(f"Number of text chunks: {len(texts)}")

# Explicitly embed the texts using OpenAIEmbeddings
embedded_texts = [embeddings.embed(text) for text in texts]

# Perform the indexing process and upload embeddings to Deep Lake
my_activeloop_org_id = os.environ["ACTIVELOOP_ORG_ID"]
my_activeloop_dataset_name = "langchain_course_twitter_algorithm"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"

# Load the dataset
db = DeepLake(dataset_path=dataset_path, embedding=embeddings)
db.add_documents(embedded_texts)  # Using the embedded texts

# Define the retriever
retriever = db.as_retriever()
retriever.search_kwargs['distance_metric'] = 'cos'
retriever.search_kwargs['fetch_k'] = 100
retriever.search_kwargs['maximal_marginal_relevance'] = True
retriever.search_kwargs['k'] = 10

# Connect to GPT for question answering
model = ChatOpenAI(model='gpt-3.5-turbo')  # switch to 'gpt-4'
qa = RetrievalQA.from_llm(model, retriever=retriever)

# Define questions and get answers
questions = [
    "What does favCountParams do?",
    "is it Likes + Bookmarks, or not clear from the code?",
]
chat_history = []

for question in questions:
    result = qa({"question": question, "chat_history": chat_history})
    chat_history.append((question, result['answer']))
    print(f"-> **Question**: {question} \n")
    print(f"**Answer**: {result['answer']} \n")
