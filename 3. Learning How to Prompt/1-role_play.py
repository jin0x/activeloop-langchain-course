from dotenv import load_dotenv
load_dotenv()

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import OpenAI

# Initialize LLM
llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)

template = """
As a futuristic robot band conductor, I need you to help me come up with a song title.
What's a cool song title for a song about {theme} in the year {year}?
"""

prompt = PromptTemplate(
    input_variables=["theme", "year"],
    template=template
)

# Input data for the prompt
input_data = {"theme": "interstellar travel", "year": "3030"}

# Create the LLMChain for the prompt
chain = LLMChain(llm=llm, prompt=prompt)

# Run the LLMChain to get the AI-generated song title
response = chain.invoke(input_data)

print("Theme: interstellar travel")
print("Year: 3030")
print("AI-generated song title:", response)
