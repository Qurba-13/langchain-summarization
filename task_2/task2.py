#configuring the azure openai mode in langchain
import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

azure_ai= AzureChatOpenAI(api_version= os.getenv("API_VERSION"),
api_key=os.getenv("AZURE_OPENAI_API_KEY"),
azure_endpoint=os.getenv("ENDPOINT_URL"),
azure_deployment=os.getenv("Chat_DEPLOYMENT_NAME"),
)


#summarize into 3 sentences
prompt_3_sentences=ChatPromptTemplate.from_template("summarize given paragraph into 3 sentences only {text}")
chain_3= prompt_3_sentences|azure_ai | StrOutputParser()
# print(chain_3.invoke({"text":ai_text})
# )

#summarize it in one sentence only
prompt_1_sentence= ChatPromptTemplate.from_template("Summarize the given paragraph into one sentence only, but it should be concise and to the point {text}")
chain_1=prompt_1_sentence | azure_ai | StrOutputParser()

if __name__=="__main__":
    ai_text="""Artificial Intelligence (AI) refers to the simulation of human intelligence by computer systems, enabling machines to perform tasks that typically require human cognition, such as learning, reasoning, problem-solving, and decision-making. Rather than just executing pre-programmed commands, AI systems use data and algorithms to identify patterns, make predictions, and improve their performance over time."""
    print("3 sentence summary is: \n")
    print(chain_3.invoke({"text":ai_text}))
    print("1 sentence summary is: \n")
    print(chain_1.invoke({"text":ai_text}))