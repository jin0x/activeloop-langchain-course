from dotenv import load_dotenv
load_dotenv()

from langchain.chains import ConversationChain, LLMChain, SimpleSequentialChain
from langchain.chains.base import Chain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.memory import ConversationBufferMemory

prompt_template = "What is a word to replace the following: {word}?"

llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)

llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(prompt_template)
)

# __call__ method
llm_chain("artificial")

# .apply() method
input_list = [
    {"word": "artificial"},
    {"word": "intelligence"},
    {"word": "robot"}
]

llm_chain.apply(input_list)

# .generate() method
response = llm_chain.generate(input_list)

print(response)

# .predict() method
prompt_template = "Looking at the context of '{context}'. What is an appropriate word to replace the following: {word}?"

llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(template=prompt_template, input_variables=["word", "context"]))

llm_chain.predict(word="fan", context="object")

# or .run() method
response = llm_chain.run(word="fan", context="object")

print(response)

