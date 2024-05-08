from dotenv import load_dotenv
load_dotenv()
import os

# Importing necessary modules
from langchain_openai import OpenAI
from langchain.agents import AgentType, load_tools, initialize_agent, Tool
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Loading the language model to control the agent
llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)

# Loading some tools to use. The llm-math tool uses an LLM, so we pass that in.
tools = load_tools(["google-search", "llm-math"], llm=llm)

# Initializing an agent with the tools, the language model, and the type of agent we want to use.
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Testing the agent
query = "What's the result of 1000 plus the number of goals scored in the soccer world cup in 2018?"
response = agent.run(query)
# print(response)

prompt = PromptTemplate(
    input_variables=["query"],
    template="You're a renowned science fiction writer. {query}"
)

llm_chain = LLMChain(llm=llm, prompt=prompt)

tools = [
    Tool(
        name='Science Fiction Writer',
        func=llm_chain.run,
        description='Use this tool for generating science fiction stories. Input should be a command about generating specific types of stories.'
    )
]

# Initializing an agent with the tools, the language model, and the type of agent we want to use.
agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Testing the agent with the new prompt
response = agent.run("Compose an epic science fiction saga about interstellar explorers")
print(response)