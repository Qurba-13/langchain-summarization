import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langgraph.prebuilt import create_react_agent

from task_2.task2 import chain_3
from task_3.task3 import retriever
load_dotenv()

@tool
def retrieved_text(query:str)->str:
    """Retrieve relevant text from the document."""
    docs=retriever.invoke(query)
    retrieve_text="\n".join([i.page_content for i in docs])
    return retrieve_text

@tool
def summarize_text(text:str)->str:
    """Summarize the retrieved text"""
    return str(chain_3.invoke({"text":text}))

@tool
def count_words(text:str)->str:
    """Count words in the summary"""
    word_count=len(text.split())
    return f"Word count:{word_count}"

tools =[retrieved_text,summarize_text,count_words]

llm = AzureChatOpenAI(
    api_version=os.getenv("API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("ENDPOINT_URL"),
    azure_deployment=os.getenv("CHAT_DEPLOYMENT_NAME"),
)

agent=create_react_agent(llm,tools)

response=agent.invoke({
    "messages":[
        (
            "user",
            "Find and summarize text about AI breakthroughs from the document. Then count the words in the summary."
        )
    ]


})
print(response["messages"][-1].content)