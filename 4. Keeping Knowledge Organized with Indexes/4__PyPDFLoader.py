
from langchain_community.document_loaders import PyPDFLoader
loader = PyPDFLoader("../data/The One Page Linux Manual.pdf")
pages = loader.load_and_split()

print(pages[0])