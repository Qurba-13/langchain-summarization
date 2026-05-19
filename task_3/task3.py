from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
load_dotenv()

azure_ai = AzureChatOpenAI(
    api_version=os.getenv("API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("ENDPOINT_URL"),
    azure_deployment=os.getenv("Chat_DEPLOYMENT_NAME"),
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Summarize the following text in 3 sentences only."),
    ("human", "{text}")
])
chain_3 = prompt | azure_ai | StrOutputParser()

file_location = "ai_intro.txt"
load_file = TextLoader(file_location, encoding="utf-8")
content = load_file.load()

text_splitting = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
documents = text_splitting.split_documents(content)
print(f"The number of chunks are {len(documents)}")

azure_embedding = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("ENDPOINT_URL"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("API_VERSION"),
    model=os.getenv("EMBEDDING_DEPLOYMENT_NAME")
)

store_vector = FAISS.from_documents(documents, azure_embedding)
retriever = store_vector.as_retriever()

query = "AI milestones"
retrieved_document = retriever.invoke(query)
retrieved_text = "\n".join([i.page_content for i in retrieved_document])

print("The summary of the text using the task2 prompt:\n")
summary = chain_3.invoke({"text": retrieved_text})
print(summary)