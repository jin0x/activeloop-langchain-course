from dotenv import load_dotenv
load_dotenv()

from langchain.chains import ConversationChain, LLMChain, SimpleSequentialChain
from langchain.chains.base import Chain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.memory import ConversationBufferMemory

llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)

## 2. Conversational Chain (Memory)
output_parser = CommaSeparatedListOutputParser()
conversation = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory()
)

conversation.predict(input="List all possible words as substitute for 'artificial' as comma separated.")

response = conversation.predict(input="And the next 4?")

print(response)