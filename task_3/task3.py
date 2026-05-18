from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv
from task_2.task2 import chain_3
load_dotenv()

file_location="ai_intro.txt"
def load_document(file_location:str):
    load_file=TextLoader(file_location,encoding="utf-8")
    return load_file.load()


# content=load_file.load()
# print(content)
def text_split(documents):
    text_splitting=RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=20)
    return text_splitting.split_documents(documents)

# documents=text_splitting.split_documents(content)
#converting the chunks into vectors
def embeddings():
    return AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("ENDPOINT_URL"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("API_VERSION"),
    model=os.getenv("EMBEDDING_DEPLOYMENT_NAME")
)
#storing the vector
def store_vector(documents):
    get_embeddings=embeddings()
    return FAISS.from_documents(documents,get_embeddings)


#retriever
def create_retriever(file_location:str):
    docs=load_document(file_location)
    split_docs=text_split(docs)
    vector_store=store_vector(split_docs)
    return vector_store.as_retriever()


retriever=create_retriever(file_location)


#testing
query="AI milestones"
retrieved_document=retriever.invoke(query)
# for i in retrieved_document:
#     print(i.page_content)

# combining all the retrived
retrieved_text="\n".join([i.page_content for i in retrieved_document])
print("The summary of the text using the task2 prompt \n")
summary=chain_3.invoke({"text": retrieved_text})
print(summary)