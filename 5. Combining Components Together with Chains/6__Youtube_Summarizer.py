import os
from dotenv import load_dotenv
import yt_dlp
import whisper
import textwrap
from langchain import OpenAI, LLMChain
from langchain_community.vectorstores import DeepLake
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains.mapreduce import MapReduceChain
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# load API kets
load_dotenv()

# set API keys
openai_api_key = os.environ['OPENAI_API_KEY']
activeloop_token = os.environ['ACTIVELOOP_TOKEN']

def download_mp4_from_youtube(urls, job_id):
    # This will hold the titles and authors of each downloaded video
    video_info = []

    for i, url in enumerate(urls):
        # Set the options for the download
        file_temp = f'./{job_id}_{i}.mp4'
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
            'outtmpl': file_temp,
            'quiet': True,
        }

        # Download the video file
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(url, download=True)
            title = result.get('title', "")
            author = result.get('uploader', "")

        # Add the title and author to our list
        video_info.append((file_temp, title, author))

    return video_info

urls=["https://www.youtube.com/watch?v=mBjPyte2ZZo&t=78s",
    "https://www.youtube.com/watch?v=cjs7QKJNVYM",]

videos_details = download_mp4_from_youtube(urls, 1)

# Whisper
model = whisper.load_model("base")

# iterate through each video and transcribe
results = []
for video in videos_details:
    result = model.transcribe(video[0])
    results.append( result['text'] )
    print(f"Transcription for {video[0]}:\n{result['text']}\n")

with open ('text.txt', 'w') as file:
    file.write(results['text'])

llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)
embeddings = OpenAIEmbeddings(model='text-embedding-ada-002')

# Summarization with LangChain
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=0, separators=[" ", ",", "\n"]
)

with open('text.txt') as f:
    text = f.read()

texts = text_splitter.split_text(text)
docs = [Document(page_content=t) for t in texts[:4]]

# Chains
chain = load_summarize_chain(llm, chain_type="map_reduce")

output_summary = chain.run(docs)
wrapped_text = textwrap.fill(output_summary, width=100)
print( chain.llm_chain.prompt.template )
print(wrapped_text)

# The "stuff" approach
prompt_template = """Write a concise bullet point summary of the following:


{text}


CONSCISE SUMMARY IN BULLET POINTS:"""

BULLET_POINT_PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])


# Initialized the summarization chain using the stuff as chain_type and the prompt above.
chain = load_summarize_chain(
    llm,
    chain_type="stuff",
    prompt=BULLET_POINT_PROMPT
)

output_summary = chain.run(docs)

wrapped_text = textwrap.fill(
    output_summary,
    width=1000,
    break_long_words=False,
    replace_whitespace=False
)
print(wrapped_text)

# The 'refine' summarization chain is a method for generating more accurate and context-aware summaries.
# This method can result in more accurate and context-aware summaries compared to other chain types like 'stuff' and 'map_reduce'.
chain = load_summarize_chain(llm, chain_type="refine")

output_summary = chain.run(docs)
wrapped_text = textwrap.fill(output_summary, width=100)
print(wrapped_text)


# create Deep Lake dataset
my_activeloop_org_id = os.environ["ACTIVELOOP_ORG_ID"]
my_activeloop_dataset_name = "langchain_course_youtube_summarizer"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"

db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)
db.add_documents(docs)

# retriever
retriever = db.as_retriever()
retriever.search_kwargs['distance_metric'] = 'cos'
retriever.search_kwargs['k'] = 4


prompt_template = """Use the following pieces of transcripts from a video to answer the question in bullet points and summarized. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {question}
Summarized answer in bullter points:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": PROMPT}
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs=chain_type_kwargs
)

print( qa.run("Summarize the mentions of google according to their AI program") )