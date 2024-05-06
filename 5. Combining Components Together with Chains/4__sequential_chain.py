from dotenv import load_dotenv
load_dotenv()

from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.chains.base import Chain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.output_parsers import CommaSeparatedListOutputParser
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

llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)

output_parser = CommaSeparatedListOutputParser()

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

overall_chain = SimpleSequentialChain(chains=[poet_chain, critic_chain])
# Run the poet and critic chain with a specific theme
theme: str = "the beauty of nature"
review = overall_chain.run(theme)

# Print the review to see the critic's evaluation
print(review)