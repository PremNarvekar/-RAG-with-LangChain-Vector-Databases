import os
from dotenv import load_dotenv

load_dotenv()

from importlib.metadata import version
from langchain_google_genai import ChatGoogleGenerativeAI

print("LangChain Core:", version("langchain-core"))
print("LangGraph:", version("langgraph"))

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API"),
)

response = llm.invoke("Hello! Introduce yourself.")
print(response.content)