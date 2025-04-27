import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from langchain_google_genai import ChatGoogleGenerativeAI

dot_env_path = Path(__file__).parents[1] / ".env"
if dot_env_path.is_file():
    load_dotenv(dotenv_path=dot_env_path)

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

embedding_client = genai.Client(api_key=GOOGLE_API_KEY, vertexai=False)

llm_client = ChatGoogleGenerativeAI(
    api_key=GOOGLE_API_KEY, model="gemini-2.0-flash-001", temperature=0, max_retries=10
)
