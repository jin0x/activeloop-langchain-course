from dotenv import load_dotenv
load_dotenv()
import os

import requests
import tqdm
from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.document_loaders import PyPDFLoader


# financial reports of amazon, but can be replaced by any URLs of PDFs
urls = [
    'https://s2.q4cdn.com/299287126/files/doc_financials/Q1_2018_-_8-K_Press_Release_FILED.pdf',
    'https://s2.q4cdn.com/299287126/files/doc_financials/Q2_2018_Earnings_Release.pdf',
    'https://s2.q4cdn.com/299287126/files/doc_news/archive/Q318-Amazon-Earnings-Press-Release.pdf',
    'https://s2.q4cdn.com/299287126/files/doc_news/archive/AMAZON.COM-ANNOUNCES-FOURTH-QUARTER-SALES-UP-20-TO-$72.4-BILLION.pdf',
    'https://s2.q4cdn.com/299287126/files/doc_financials/Q119_Amazon_Earnings_Press_Release_FINAL.pdf',
    'https://s2.q4cdn.com/299287126/files/doc_news/archive/Amazon-Q2-2019-Earnings-Release.pdf',
    'https://s2.q4cdn.com/299287126/files/doc_news/archive/Q3-2019-Amazon-Financial-Results.pdf',
    'https://s2.q4cdn.com/299287126/files/doc_news/archive/Amazon-Q4-2019-Earnings-Release.pdf',
    'https://s2.q4cdn.com/299287126/files/doc_financials/2020/Q1/AMZN-Q1-2020-Earnings-Release.pdf',
    'https://s2.q4cdn.com/299287126/files/doc_financials/2020/q2/Q2-2020-Amazon-Earnings-Release.pdf',
    'https://s2.q4cdn.com/299287126/files/doc_financials/2020/q4/Amazon-Q4-2020-Earnings-Release.pdf',
    'https://s2.q4cdn.com/299287126/files/doc_financials/2021/q1/Amazon-Q1-2021-Earnings-Release.pdf',
    'https://s2.q4cdn.com/299287126/files/doc_financials/2021/q2/AMZN-Q2-2021-Earnings-Release.pdf',
    'https://s2.q4cdn.com/299287126/files/doc_financials/2021/q3/Q3-2021-Earnings-Release.pdf',
    'https://s2.q4cdn.com/299287126/files/doc_financials/2021/q4/business_and_financial_update.pdf',
    'https://s2.q4cdn.com/299287126/files/doc_financials/2022/q1/Q1-2022-Amazon-Earnings-Release.pdf',
    'https://s2.q4cdn.com/299287126/files/doc_financials/2022/q2/Q2-2022-Amazon-Earnings-Release.pdf',
    'https://s2.q4cdn.com/299287126/files/doc_financials/2022/q3/Q3-2022-Amazon-Earnings-Release.pdf',
    'https://s2.q4cdn.com/299287126/files/doc_financials/2022/q4/Q4-2022-Amazon-Earnings-Release.pdf'
]

def load_reports(urls: List[str]) -> List[str]:
    """ Load pages from a list of urls"""
    pages = []

    for url in tqdm.tqdm(urls):
        r = requests.get(url)
        path = url.split('/')[-1]
        with open(path, 'wb') as f:
            f.write(r.content)
        loader = PyPDFLoader(path)
        local_pages = loader.load_and_split()
        pages.extend(local_pages)
    return pages

pages = load_reports(urls)


text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(pages)

# embeddings = OpenAIEmbeddings()
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

my_activeloop_org_id = os.environ["ACTIVELOOP_ORG_ID"]
my_activeloop_dataset_name = "amazon_earnings"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"

db = DeepLake(dataset_path=dataset_path, embedding=embeddings)
db.add_documents(texts)

qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model='gpt-3.5-turbo'),
    chain_type='stuff',
    retriever=db.as_retriever()
)

qa.invoke("Combine total revenue in 2020?")
