from dotenv import load_dotenv
load_dotenv()

from langchain.chains import ConversationChain, LLMChain, SimpleSequentialChain
from langchain.chains.base import Chain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.memory import ConversationBufferMemory

prompt_template = "What is a word to replace the following: {word}?"

# Set the "OPENAI_API_KEY" environment variable before running following line.
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
llm_chain.generate(input_list)


# .predict() method
prompt_template = "Looking at the context of '{context}'. What is an appropriate word to replace the following: {word}?"

llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(template=prompt_template, input_variables=["word", "context"]))

llm_chain.predict(word="fan", context="object")

# or .run() method
llm_chain.run(word="fan", context="object")

## 1. Parsers
output_parser = CommaSeparatedListOutputParser()
template = """List all possible words as substitute for 'artificial' as comma separated."""

llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate(template=template, output_parser=output_parser, input_variables=[]),
    output_parser=output_parser)

llm_chain.predict()

## 2. Conversational Chain (Memory)
output_parser = CommaSeparatedListOutputParser()
conversation = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory()
)

conversation.predict(input="List all possible words as substitute for 'artificial' as comma separated.")

conversation.predict(input="And the next 4?")

## 3. Sequential Chain
# poet
poet_template: str = """You are an American poet, your job is to come up with\
poems based on a given theme.

Here is the theme you have been asked to generate a poem on:
{input}\
"""

poet_prompt_template: PromptTemplate = PromptTemplate(
    input_variables=["input"], template=poet_template)

# creating the poet chain
poet_chain: LLMChain = LLMChain(
    llm=llm, output_key="poem", prompt=poet_prompt_template)

# critic
critic_template: str = """You are a critic of poems, you are tasked\
to inspect the themes of poems. Identify whether the poem includes romantic expressions or descriptions of nature.

Your response should be in the following format, as a Python Dictionary.
poem: this should be the poem you received
Romantic_expressions: True or False
Nature_descriptions: True or False

Here is the poem submitted to you:
{poem}\
"""

critic_prompt_template: PromptTemplate = PromptTemplate(
    input_variables=["poem"], template=critic_template)

# creating the critic chain
critic_chain: LLMChain = LLMChain(
    llm=llm, output_key="critic_verified", prompt=critic_prompt_template)


## 4. Debug
template = """List all possible words as substitute for 'artificial' as comma separated.

Current conversation:
{history}

{input}"""

conversation = ConversationChain(
    llm=llm,
    prompt=PromptTemplate(template=template, input_variables=["history", "input"], output_parser=output_parser),
    memory=ConversationBufferMemory(),
    verbose=True)

conversation.predict(input="")

##  5. Custom Chain
from typing import Dict, List


class ConcatenateChain(Chain):
    chain_1: LLMChain
    chain_2: LLMChain

    @property
    def input_keys(self) -> List[str]:
        # Union of the input keys of the two chains.
        all_input_vars = set(self.chain_1.input_keys).union(set(self.chain_2.input_keys))
        return list(all_input_vars)

    @property
    def output_keys(self) -> List[str]:
        return ['concat_output']

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        output_1 = self.chain_1.run(inputs)
        output_2 = self.chain_2.run(inputs)
        return {'concat_output': output_1 + output_2}

prompt_1 = PromptTemplate(
    input_variables=["word"],
    template="What is the meaning of the following word '{word}'?",
)
chain_1 = LLMChain(llm=llm, prompt=prompt_1)

prompt_2 = PromptTemplate(
    input_variables=["word"],
    template="What is a word to replace the following: {word}?",
)
chain_2 = LLMChain(llm=llm, prompt=prompt_2)

concat_chain = ConcatenateChain(chain_1=chain_1, chain_2=chain_2)
concat_output = concat_chain.run("artificial")
print(f"Concatenated output:\n{concat_output}")