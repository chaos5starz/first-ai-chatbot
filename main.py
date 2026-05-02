from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from datetime import datetime
import gradio as gr

load_dotenv()

def getdate():
    """get the current date"""
    return datetime.now().strftime("%Y-%m-%d")

llm = ChatOllama(model="qwen2.5:3b")

system_prompt = """
you are a helpful assistant that can answer user questions,
use the getdate function to get the current date."""

agent = create_agent(model = llm ,tools = [getdate] , system_prompt = system_prompt)


def chat(user_query , history):
    response = agent.invoke({"messages": [{"role": "user", "content": user_query}]})
    last_respone = response["messages"][-1].content
    return last_respone


with gr.Blocks() as demo :
    gr.Markdown("Ai chatbot")
    gr.ChatInterface(fn = chat)

    
demo.launch()



