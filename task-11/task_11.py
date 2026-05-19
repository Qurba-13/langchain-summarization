import os
from datetime import datetime
from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')
from langchain_core.tools import tool
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.prebuilt import create_react_agent
azure_ai = AzureChatOpenAI(
    api_version=os.getenv("API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("ENDPOINT_URL"),
    azure_deployment=os.getenv("Chat_DEPLOYMENT_NAME"),
)
summarization_prompt=ChatPromptTemplate.from_messages([
    ("system", "you are a helpful assistant. Summarize the given text in 3 sentences only."),
    ("human","{text}")
])
summarization_chain=summarization_prompt|azure_ai|StrOutputParser()

def summarize_text(text:str)->str:
    return summarization_chain.invoke({"text":text})

def get_date()->str:
    today_date=datetime.now().strftime("%Y-%m-%d")
    return f"Todays date is {today_date}"
def mock_web_search(query: str) -> str:
    return """AI trends in 2026 include rapid advancements in multimodal models 
that can process text, images, and audio simultaneously. Large language models 
are becoming more efficient and accessible. Governments worldwide are introducing 
AI regulations to ensure ethical use. AI is increasingly being used in healthcare 
for drug discovery and diagnostics. Autonomous agents are emerging as the next 
frontier in AI development."""
def count_words(text:str)->str:
    word_count=len(text.split())
    return f"Word count is {word_count}"

@tool
def tool_summarize(text:str)->str:
    """This tool summarizes the text"""
    return summarize_text(text)
@tool
def get_current_date()->str:
    """"This tool gets todays date"""
    return get_date()
@tool
def tool_web_search(query:str)->str:
    """Tis tool return placeholder text"""
    return mock_web_search(query)

@tool
def word_count_text(text:str)->str:
    """This tool return the count of the returned result"""

    return count_words(text)

tools_list=[tool_summarize,get_current_date,tool_web_search,word_count_text]
agent=create_react_agent(azure_ai,tools_list)
print("Summarize the text and get todays date \n")

ai_text_placeholde="""Artificial intelligence is expected to significantly shape American society over the next 20 years, but opinions differ between experts and the public. Most AI experts believe AI will positively impact the economy, jobs, health care, and education, while the public remains more cautious and concerned about negative effects. Gender differences are also notable, with male experts generally more optimistic than female experts. Both experts and the public agree that AI may negatively affect elections and news due to misinformation and manipulation risks. Overall, experts are more confident in AI’s benefits, whereas the public remains uncertain about whether AI will ultimately improve or harm society.
"""
response1=agent.invoke({
    "messages":[("user",f"Summarize the given text {ai_text_placeholde} and also tell me todays date. \n")]

})
#summarize the ai trends and also mock web searcg
response2=agent.invoke({
    "messages":[("user",f"Summarize Ai trends and search for recent updates" )]
})
print(response1["messages"][-1].content)
print(response2["messages"][-1].content)