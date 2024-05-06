from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel, Field, field_validator
from typing import List
from langchain.output_parsers import PydanticOutputParser, RetryWithErrorOutputParser, OutputFixingParser
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI

# Define your desired data structure
class Suggestions(BaseModel):
    words: List[str] = Field(description="List of substitute words based on given context")
    reasons: List[str] = Field(description="the reasoning of why this word fits the context")

    # Throw error in case of receiving a numbered-list from API
    @field_validator('words')
    @classmethod
    def not_start_with_number(cls, field):
        for item in field:
            if item[0].isnumeric():
                raise ValueError("The word cannot start with a number")
        return field

parser = PydanticOutputParser(pydantic_object=Suggestions)

template = """
Offer a list of suggestions to substitute the specified target_word based the presented context.
{format_instructions}
target_word={target_word}
context={context}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["target_word", "context"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

model_input = prompt.format_prompt(
    target_word="behavior",
    context="The behavior of the students in the classroom was disruptive and made it difficult for the teacher to conduct the lesson."
)

model = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)

output = model(model_input.to_string())
response = parser.parse(output)
# print(response)

missformatted_output = '{"words": ["conduct", "manner"], "reasoning": ["refers to the way someone acts in a particular situation.", "refers to the way someone behaves in a particular situation."]}'
outputfixing_parser = OutputFixingParser.from_llm(parser=parser, llm=model)
response = outputfixing_parser.parse(missformatted_output)
# print(response)

retry_missformatted_output = '{"words": ["conduct", "manner"]}'
retry_parser = RetryWithErrorOutputParser.from_llm(parser=parser, llm=model)
response =retry_parser.parse_with_prompt(retry_missformatted_output, model_input)
# print(response)
