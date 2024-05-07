from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate

llm = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template("The following is a friendly conversation between a human and an AI."),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{input}")
])

# No memory retention
# conversation = ConversationChain(llm=llm, verbose=True)
# output = conversation.predict(input="Hi there!")
# output = conversation.predict(input="In what scenarios extra memory should be used?")
# output = conversation.predict(input="There are various types of memory in Langchain. When to use which type?")
# output = conversation.predict(input="Do you remember what was our first message?")
# print(output)

# Memory save context
# memory.save_context({"input": "hi there!"}, {"output": "Hi there! It's nice to meet you. How can I help you today?"})
# print( memory.load_memory_variables({}) )

# Memory retention
memory = ConversationBufferMemory(return_messages=True)
conversation = ConversationChain(llm=llm, memory=memory, prompt=prompt, verbose=True)
# print( conversation.predict(input="Tell me a joke about elephants") )
# print( conversation.predict(input="Who is the author of the Harry Potter series?") )
# print( conversation.predict(input="What was the joke you told me earlier?") )


# Start a general question
user_message = "Tell me about the history of the Internet."
response = conversation(user_message)
print(response)
user_message = "Who are some important figures in its development?"
response = conversation(user_message)
print(response)
user_message = "What did Tim Berners-Lee contribute?"
response = conversation(user_message)
print(response)