from langchain_openai import OpenAI
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.tools import tool, BaseTool, StructuredTool
from langchain_community.vectorstores.pinecone import Pinecone

from db_helper import get_df, init_pinecone
from tool_helper import get_price_change_tool

from dotenv import load_dotenv


load_dotenv()

# prepare data
df = get_df()

# init index
index = init_pinecone()

