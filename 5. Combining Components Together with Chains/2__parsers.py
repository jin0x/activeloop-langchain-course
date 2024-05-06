from dotenv import load_dotenv
load_dotenv()

from langchain.chains import ConversationChain, LLMChain, SimpleSequentialChain
from langchain.chains.base import Chain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.memory import ConversationBufferMemory

llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)

## 1. Parsers
output_parser = CommaSeparatedListOutputParser()
template = """List all possible words as substitute for 'artificial' as comma separated."""

llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(template=template, output_parser=output_parser, input_variables=[]),
    output_parser=output_parser)

response = llm_chain.predict()

print(response)