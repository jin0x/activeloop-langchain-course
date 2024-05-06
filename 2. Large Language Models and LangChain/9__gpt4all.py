from dotenv import load_dotenv
load_dotenv()

# Import modules
from langchain.llms import GPT4All
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

model_path = 'the_path_to_your_local_model'
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = GPT4All(model=model_path, callback_manager=callback_manager, verbose=True)

template = """Question: {question}

Answer: Let's think step by step."""

prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt, llm=llm)
query = "What happens when it rains somewhere?"

result = llm_chain.run(query)

print(result)