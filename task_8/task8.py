import os
from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
azure_ai = AzureChatOpenAI(
    api_version=os.getenv("API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("ENDPOINT_URL"),
    azure_deployment=os.getenv("Chat_DEPLOYMENT_NAME"),
)

class SummaryOutput(BaseModel):
    summary:str = Field(description="3 sentence summary of the text")
    length: int = Field(description="number of the characters in the summary")

output_structred=JsonOutputParser(pydantic_object=SummaryOutput)

format_instructions=output_structred.get_format_instructions()
print(format_instructions)

#prompts
prompt=ChatPromptTemplate.from_messages(
    [("system","""You are a assistant which summarizes the text. and you must give the output in the following format: {format_instructions}"""),
    ("human","Summarize this text in 3 senetence:{text}")

    ]
)
chain=prompt|azure_ai|output_structred

ai_text = """Artificial intelligence is transforming numerous industries 
by automating complex tasks and enabling smarter decision-making. 
In healthcare, AI helps doctors detect diseases earlier and more accurately 
through advanced image recognition. In finance, it powers fraud detection 
systems that analyze millions of transactions in real time. 
Education is also being revolutionized, with AI tutors personalizing 
learning experiences for individual students. Meanwhile, in transportation, 
self-driving vehicles are becoming a reality thanks to AI breakthroughs. 
These applications demonstrate how AI is reshaping the world across 
every sector, making processes faster, cheaper, and more efficient 
than ever before."""
result=chain.invoke({
    "format_instructions":format_instructions,
    "text":ai_text
})
print(type(result))
print(result)