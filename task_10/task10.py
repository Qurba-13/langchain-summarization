import os
from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')

from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader

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

#question answers
qanda_prompt=ChatPromptTemplate.from_messages([
    ("system","You are a helpful assistant. Answer the questions based on the given text only."),
    ("human", "Text:\n{text}\n\n Question:{question}")

])
qanda_chain=qanda_prompt|azure_ai|StrOutputParser()

def load_document(file_location):
    loader=TextLoader(file_location,encoding="utf-8")
    docs=loader.load()
    full_text="\n".join([i.page_content for i in docs])
    return full_text

def summarize_text(text):
    summary=summarization_chain.invoke({"text":text})
    return summary
def answer_questions(text,question):
    answer=qanda_chain.invoke({"text":text,"question":question})
    return answer
def compare_answers(question,summary,full_text):
    print(f"Question: {question}\n")
    answer_from_summary=answer_questions(summary,question)
    print(f"The answer from the summary is {answer_from_summary} \n")
    answer_from_full=answer_questions(full_text,question)
    print(f"Answer from full text is: {answer_from_full} \n")
    return answer_from_summary,answer_from_full

def judge_answers(question, answer_summary, answer_full):
    judge_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert evaluator. Compare two answers to the same question.
        Judge which is more concise and accurate.
        At the end clearly state: 'WINNER: Summary' or 'WINNER: Full Document' and explain why in 2 sentences."""),
        ("human", """
        Question: {question}
        Answer 1 (from Summary):
        {answer_summary}
        Answer 2 (from Full Document):
        {answer_full}
        Which answer is better?""") ])
    judge_chain = judge_prompt | azure_ai | StrOutputParser()
    verdict = judge_chain.invoke({
        "question": question,
        "answer_summary": answer_summary,
        "answer_full": answer_full
    })
    return verdict

if __name__== "__main__":
    full_text=load_document("ai_intro.txt")
    summary=summarize_text(full_text)
    question="What is the key event mentioned?"
    answer_from_summary, answer_from_full=compare_answers(question,summary,full_text)
    verdict=judge_answers(question,answer_from_summary,answer_from_full)
    print(verdict)
