from dotenv import load_dotenv
import os


(load_dotenv(dotenv_path=".env"))
api_version= os.getenv("API_VERSION")
api_key=os.getenv("AZURE_OPENAI_API_KEY")
api_endpoint_url=os.getenv("ENDPOINT_URL")
api_deployment_name=os.getenv("Chat_DEPLOYMENT_NAME")

print(api_version)
print(api_key)
print(api_endpoint_url)
print(api_deployment_name)