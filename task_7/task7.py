import os
from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')
from langchain_openai import AzureChatOpenAI,AzureOpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
load_dotenv()
azure_ai= AzureChatOpenAI(api_version= os.getenv("API_VERSION"),
api_key=os.getenv("AZURE_OPENAI_API_KEY"),
azure_endpoint=os.getenv("ENDPOINT_URL"),
azure_deployment=os.getenv("Chat_DEPLOYMENT_NAME")
)

#embeddings
azure_embeddings=AzureOpenAIEmbeddings(
    api_version= os.getenv("API_VERSION"),
api_key=os.getenv("AZURE_OPENAI_API_KEY"),
azure_endpoint=os.getenv("ENDPOINT_URL"),
model=os.getenv("EMBEDDING_DEPLOYMENT_NAME")

)
#prompt for summarizarion of webpage, and pdf
prompt=ChatPromptTemplate.from_messages([
    ("system","You are an assistant. Summaeize the following text in 3 sentences."),
    ("human","{text}")
])
chain=prompt | azure_ai|StrOutputParser()
#load the pdf:
pdf_loader=PyPDFLoader("ethics.pdf ")
pdf_doc=pdf_loader.load()
print(f"The lengtth of the pdf is {len(pdf_doc)} pages")
#breaking the pdf into chunks
setting_chunks=RecursiveCharacterTextSplitter(chunk_size=150,chunk_overlap=30)
pdf_chunks=setting_chunks.split_documents(pdf_doc)
#number of chunks in the document
print(f"The chunks in the documents are {len(pdf_chunks) } chunks")
#convert the chunks into numbers -> vector and using fais to store them
pdf_vector_store=FAISS.from_documents(pdf_chunks,azure_embeddings)

#webscraping of the webpage
web_loader=WebBaseLoader("https://builtin.com/artificial-intelligence")
web_content=web_loader.load()
print(f"The length of the webpage is {len(web_content)}")
#splitting the webpage
web_split=setting_chunks.split_documents(web_content)
#number of chunks 
print(f"The chunks in the webpage is {len(web_split)}")
#convert it into vector
webpage_chunks=FAISS.from_documents(web_split,azure_embeddings)

#query
query="AI challenges"
pdf_query_retrieve=pdf_vector_store.as_retriever(search_kwargs={"k":3})
results_pdf=pdf_query_retrieve.invoke(query)
join_pdf=" ".join([i.page_content for i in results_pdf])
pdf_summary=chain.invoke({"text":join_pdf})

#run the query on the webpage
web_query_retrieve=webpage_chunks.as_retriever(search_kwargs={"k":3})
results_webpage=web_query_retrieve.invoke(query)
join_web=" ".join([j.page_content for j in results_webpage])
web_summary=chain.invoke({"text":join_web})

print("The result for the summary of pdf is:\n")
print(pdf_summary)
print("\n")
print("The results for the summary of the webpage is: \n")
print(web_summary)