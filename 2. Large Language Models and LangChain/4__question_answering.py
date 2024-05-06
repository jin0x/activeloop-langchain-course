from dotenv import load_dotenv
load_dotenv()

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.chat_models.huggingface import HuggingFaceEndpoint

template = """Question: {question}

Answer: """
prompt = PromptTemplate(
    template=template,
    input_variables=['question']
)

# user question
question =input("")

# initialize Hub LLM
hub_llm = HuggingFaceEndpoint(
    repo_id='google/flan-t5-large',
    task="text-generation",
    temperature=0,
    # model_kwargs={
    #     "max_new_tokens": 512,
    #     "top_k": 30,
    #     "temperature": 0,
    #     "repetition_penalty": 1.03,
    # },
)

# create prompt template > LLM chain
llm_chain = LLMChain(
    llm=hub_llm,
    prompt=prompt,
    verbose=True
)

# ask the user question about the capital of France
print(llm_chain.run(question))