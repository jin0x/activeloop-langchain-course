from dotenv import load_dotenv
load_dotenv()

import os

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import DeepLake
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from dotenv import load_dotenv

load_dotenv()


# text to write to a local file
# taken from https://www.theverge.com/2023/3/14/23639313/google-ai-language-model-palm-api-challenge-openai
text = """Google opens up its AI language model PaLM to challenge OpenAI and GPT-3
Google is offering developers access to one of its most advanced AI language models: PaLM.
The search giant is launching an API for PaLM alongside a number of AI enterprise tools
it says will help businesses “generate text, images, code, videos, audio, and more from
simple natural language prompts.”

PaLM is a large language model, or LLM, similar to the GPT series created by OpenAI or
Meta’s LLaMA family of models. Google first announced PaLM in April 2022. Like other LLMs,
PaLM is a flexible system that can potentially carry out all sorts of text generation and
editing tasks. You could train PaLM to be a conversational chatbot like ChatGPT, for
example, or you could use it for tasks like summarizing text or even writing code.
(It’s similar to features Google also announced today for its Workspace apps like Google
Docs and Gmail.)
"""

# write text to local file
with open("my_file.txt", "w") as file:
    file.write(text)

# use TextLoader to load text from local file
loader = TextLoader("my_file.txt")
docs_from_file = loader.load()

# create a text splitter
text_splitter = CharacterTextSplitter(chunk_size=400, chunk_overlap=20)

# split documents into chunks
docs = text_splitter.split_documents(docs_from_file)

# Before executing the following code, make sure to have
# your OpenAI key saved in the “OPENAI_API_KEY” environment variable.
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

# create Deep Lake dataset
# TODO: use your organization id here. (by default, org id is your username)
my_activeloop_org_id = os.environ["ACTIVELOOP_ORG_ID"]
my_activeloop_dataset_name = "langchain_course_indexers_retrievers"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path, embedding=embeddings, overwrite=True)

# add documents to our Deep Lake dataset
db.add_documents(docs)

# create retriever from db
retriever = db.as_retriever()

# create a retrieval chain
qa_chain = RetrievalQA.from_chain_type(
	llm=OpenAI(model="gpt-3.5-turbo-instruct"),
	chain_type="stuff",
	retriever=retriever
)

query = "How Google plans to challenge OpenAI?"
response = qa_chain.invoke(query)
print(response)

# create GPT3 wrapper
llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)

# create compressor for the retriever
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
	base_compressor=compressor,
	base_retriever=retriever
)

# retrieving compressed documents
retrieved_docs = compression_retriever.invoke(query)
print(retrieved_docs[0].page_content)