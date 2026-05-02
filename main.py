import uuid
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from datetime import datetime
import gradio as gr
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3


load_dotenv()

def getdate():
    """get the current date"""
    return datetime.now().strftime("%Y-%m-%d")

llm = ChatOllama(model="qwen2.5:3b")
conn = sqlite3.connect("chatbotmemory.db" , check_same_thread=False)
checkpointer = SqliteSaver(conn)

system_prompt = """
you are a helpful assistant that can answer user questions
"""

agent = create_agent(model = llm ,
                     tools = [getdate] ,
                       system_prompt = system_prompt ,
                         checkpointer = checkpointer)

def chat(user_query , history , thread_id):
    config = {"configurable" : {"thread_id" : thread_id}}
    response = agent.invoke({"messages": [{"role": "user", "content": user_query}]} , config = config)
    last_respone = response["messages"][-1].content
    return last_respone


with gr.Blocks() as demo :
    gr.Markdown("Ai chatbot")
    thread_id = gr.State(value = lambda: str(uuid.uuid4()))
    gr.ChatInterface(fn = chat , additional_inputs=[thread_id])

    
demo.launch()



