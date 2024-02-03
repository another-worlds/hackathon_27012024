from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.tools import tool, BaseTool, StructuredTool
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings

from db_helper import get_df
from tool_helper import get_price_change_tool

from dotenv import load_dotenv


load_dotenv()

# populate database with crypto ticker entries
v