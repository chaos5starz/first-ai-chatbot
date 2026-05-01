from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from datetime import datetime


load_dotenv()

def getdate():
    """get the current date"""
    return datetime.now().strftime("%Y-%m-%d")

llm = ChatOllama(model="qwen2.5:3b")

system_prompt = """
you are a helpful assistant that can answer questions and perform tasks.
use the getdate function to get the current date."""

agent = create_agent(model = llm ,tools = [getdate] , system_prompt = system_prompt)
user_query = input("Enter a query: ")
response = agent.invoke({"messages": [{"role": "user", "content": user_query}]})

print(response["messages"][-1].content)



