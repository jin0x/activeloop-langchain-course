from dotenv import load_dotenv
load_dotenv()

from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

llm = OpenAI(model="gpt-3.5-turbo-instruct", n=2, best_of=2)

with get_openai_callback() as cb:
    result = llm("Tell me a joke")
    cost = cb.total_cost
    print(result)
    print("$",round(cost, 5))