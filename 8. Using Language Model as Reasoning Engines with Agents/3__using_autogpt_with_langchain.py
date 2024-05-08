from dotenv import load_dotenv
load_dotenv()
import os

# Importing necessary modules
import faiss
from langchain.agents import Tool
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool
from langchain_community.docstore import InMemoryDocstore
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain_experimental.autonomous_agents import AutoGPT

#Set up the tools
search = GoogleSearchAPIWrapper()
tools = [
    Tool(
        name = "search",
        func=search.run,
        description="Useful for when you need to answer questions about current events. You should ask targeted questions",
        return_direct=True
    ),
    WriteFileTool(),
    ReadFileTool(),
]


embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
embedding_size = 1536

index = faiss.IndexFlatL2(embedding_size)
vectorstore = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})

agent = AutoGPT.from_llm_and_tools(
    ai_name="Jim",
    ai_role="Assistant",
    tools=tools,
    llm=llm,
    memory=vectorstore.as_retriever()
)

# Set verbose to be true
agent.chain.verbose = True

task = "Provide an analysis of the major historical events that led to the French Revolution"

agent.run([task])