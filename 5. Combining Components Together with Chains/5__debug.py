from dotenv import load_dotenv
load_dotenv()

from langchain.chains import ConversationChain, LLMChain
from langchain.chains.base import Chain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.memory import ConversationBufferMemory
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