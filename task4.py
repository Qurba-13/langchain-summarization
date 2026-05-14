
import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langchain.agents import create_agent

from task2 import chain_3

load_dotenv()

@tool
def summarize_text(text: str) -> str:
    """Summarize the input text using the chain_3 summarizer."""
    return str(chain_3.invoke({"text": text}))

tools = [summarize_text]

llm = AzureChatOpenAI(
    api_version=os.getenv("API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("ENDPOINT_URL"),
    azure_deployment=os.getenv("CHAT_DEPLOYMENT_NAME"),
)

agent = create_agent(llm, tools)

result = agent.invoke({
    "messages": [
        ("user", "Summarize AI impact on healthcare in diagnosis, imaging, and second opinions.")
    ]
})

print(result["messages"][-1].content)
# running it again:
run2=agent.invoke({
    "messages":[("user", "Summarize something interesting")]

})
print("Vague results: \n")
print(run2["messages"][-1])