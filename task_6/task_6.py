import os
from dotenv import load_dotenv
# from langchain.chains import ConversationChain 
load_dotenv(dotenv_path='.env')
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

azure_ai= AzureChatOpenAI(api_version= os.getenv("API_VERSION"),
api_key=os.getenv("AZURE_OPENAI_API_KEY"),
azure_endpoint=os.getenv("ENDPOINT_URL"),
azure_deployment=os.getenv("Chat_DEPLOYMENT_NAME"),
)

ml_placeholder="""Machine learning (ML) is a dynamic branch of artificial intelligence that empowers computers to learn and evolve without explicit programming. Instead of following rigid, pre-written rules, ML algorithms analyze vast datasets to detect underlying patterns, self-correct, and make highly accurate predictions.

It drives much of our modern digital experience—fueling personalized recommendations on Netflix, powering autonomous vehicles, detecting fraudulent credit card transactions, and enabling facial recognition. As data continues to grow exponentially, machine learning serves as the critical engine transforming raw information into actionable intelligence, fundamentally reshaping how industries operate and solve complex global challenges."""

dl_placeholder="""Data learning—often referred to as machine learning—is the transformative process where algorithms analyze massive datasets to uncover hidden patterns, trends, and correlations. Instead of relying on manual human programming, these systems ingest raw data, recognize complex structures, and dynamically adapt their logic over time.

This continuous feedback loop turns static information into predictive power. Today, data learning drives everything from predictive healthcare diagnostics and algorithmic financial trading to smart energy grids and predictive supply chains. Ultimately, it bridges the gap between chaotic, unstructured data and intelligent, automated decision-making, serving as the cornerstone of the modern algorithmic economy."""

prompt_given=ChatPromptTemplate.from_messages([
    ("system","You are a helpful assitant that summarizes text in 3 sentences only."),
    MessagesPlaceholder(variable_name="history"),
    ("human","{text}")

])
chain=prompt_given | azure_ai | StrOutputParser()

#store last 3 interactions -> history
buffer_history=[]
ml_summary=chain.invoke({"history":buffer_history,
"text":ml_placeholder})
buffer_history.append(HumanMessage(content=ml_placeholder))
buffer_history.append(AIMessage(content=ml_summary))
buffer_history=buffer_history[-6:]

#DL text summarization
dl_summary_buffer=chain.invoke({
    "history":buffer_history,
    "text":dl_placeholder
})
#whatever we got in the history and tranform it into summary
summarize_history_prompt=ChatPromptTemplate.from_messages([
    ("system","Summarize the following conversation into 2 sentences only"),
    ("human", "{conversation}")

])

summarize_history_chain=summarize_history_prompt | azure_ai |StrOutputParser()

summary_history=[]
print("ML summary: \n")
ml_summary_2=chain.invoke({
    "history":summary_history,
    "text":ml_placeholder
})
print(ml_summary_2)

initial_convo=f"Human: {ml_placeholder}\nAI: {ml_summary_2}"
summarised_initial=summarize_history_chain.invoke({"conversation": initial_convo})
history_summary=[AIMessage(content=f"Previous conversation summary: {summarised_initial}")]

#now do dl summary
dl_summary=chain.invoke({
    "history": history_summary,
    "text": dl_placeholder
})

print("\nBUFFER MEMORY:")
print("ML Summary:\n", ml_summary)
print("\nDL Summary:\n", dl_summary_buffer)

print("\nSUMMARY MEMORY:")
print("ML Summary:\n", ml_summary_2)
print("\nDL Summary:\n", dl_summary)
