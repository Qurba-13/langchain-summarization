from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv
from task2 import chain_3
load_dotenv()

file_location="ai_intro.txt"
load_file=TextLoader(file_location,encoding="utf-8")
content=load_file.load()
# print(content)
text_splitting=RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=20)
documents=text_splitting.split_documents(content)
#converting the chunks into vectors
azure_embedding= AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("ENDPOINT_URL"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("API_VERSION"),
    model=os.getenv("EMBEDDING_DEPLOYMENT_NAME")
)
#storing the vector
store_vector=FAISS.from_documents(documents,azure_embedding)
#retriever
retriever=store_vector.as_retriever()
query="AI milestones"
retrieved_document=retriever.invoke(query)
# for i in retrieved_document:
#     print(i.page_content)

# combining all the retrived
retrieved_text="\n".join([i.page_content for i in retrieved_document])
print("The summary of the text using the task2 prompt \n")
summary=chain_3.invoke({"text": retrieved_text})
print(summary)