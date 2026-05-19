import os
from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')

from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# from langchain_community.retrievers import mu

azure_ai = AzureChatOpenAI(
    api_version=os.getenv("API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("ENDPOINT_URL"),
    azure_deployment=os.getenv("Chat_DEPLOYMENT_NAME"),
)

azure_embeddings = AzureOpenAIEmbeddings(
    api_version=os.getenv("API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("ENDPOINT_URL"),
    model=os.getenv("EMBEDDING_DEPLOYMENT_NAME"),
)

#task2 
prompt=ChatPromptTemplate.from_messages([
    ("system", "you are a helpful assistant. Summarize the given text in 3 sentences only "),
    ("human", "{text}")

])
chain=prompt|azure_ai|StrOutputParser()
#sample 500 word text 

loader=TextLoader("ai_intro.txt", encoding="utf-8")
docs=loader.load()
#splittling
splitter=RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks=splitter.split_documents(docs)
storing_vector=FAISS.from_documents(chunks,azure_embeddings)
single_prompt_retreiver=storing_vector.as_retriever(search_kwargs={"k":3})
query="AI advancement"
single_result=single_prompt_retreiver.invoke(query)
combining_result_single="\n".join([i.page_content for i in single_result])
single_result_summary=chain.invoke({"text":combining_result_single})
print(single_result_summary)
print("Multi-query result: \n")
query_three_prompt=ChatPromptTemplate.from_messages([
    ("system","""you are a helpful assistant. I will give you single prompt, you have to generate three Vversion of given prompt from the one i gave but they all should not be exactly same. Also, give each query in three different lines no extra numbers, symbols"""),
    ("human","{query}")
])
mult_query_chain=query_three_prompt|azure_ai|StrOutputParser()
alternate_text_query=  mult_query_chain.invoke({"query":query})
alternate_query=alternate_text_query.strip().split("\n")
alternate_query.append(query)
print("The generated prompts are: \n")
for i in alternate_query:
    print(f"{i} \n")

all_results=[]
for j in alternate_query:
    results=single_prompt_retreiver.invoke(j)
    all_results.extend(results)
# we have to make sure there is no duplicates
unqiue_output=list({i.page_content: i for i in all_results}.values())
combining_outputs="\n".join([j.page_content for j in unqiue_output])
print(f"Total unique chunks {len(unqiue_output)}")
multiple_summary=chain.invoke({"text":combining_outputs})
print("The multiple query summary is\n",multiple_summary)

#AI judge
judging = ChatPromptTemplate.from_messages([
    ("system", """You are an expert evaluator. Compare these two summaries and judge which one is better.
Consider: depth, detail, completeness, and informativeness.
At the end clearly state: 'WINNER: Single Query' or 'WINNER: Multi Query' and explain why in 2 sentences."""),
    ("human", """
Summary 1 (Single Query):
{single_summary}
Summary 2 (Multi Query):
{multi_summary} Which summary is better and why?""")])

judge_chain = judging | azure_ai | StrOutputParser()

better_prompt = judge_chain.invoke({
    "single_summary": single_result_summary,
    "multi_summary": multiple_summary
})

print(better_prompt)