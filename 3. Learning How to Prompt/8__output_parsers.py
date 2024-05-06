from dotenv import load_dotenv
load_dotenv()

from pydantic import BaseModel, Field, validator
from typing import List
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate

# Define your desired data structure
class Suggestions(BaseModel):
    words: List[str] = Field(description="List of substitute words based on given context")

    # Throw error in case of receiving a numbered-list from API
    @validator('words')
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

model_input = prompt.format_input(
    target_word="behavior",
    context="The behavior of the students in the classroom was disruptive and made it difficult for the teacher to conduct the lesson."
)