from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.tools import tool, BaseTool, StructuredTool

from dotenv import load_dotenv


load_dotenv()

def langchain_agent():
    llm = OpenAI(temperature=0.7)
    tools = load_tools(["wikipedia", "llm-math"], llm=llm)
    
    agent = initialize_agent(
        tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    
    result = agent.run(
        "What is the average height of a dog? Multiply it by average dog age."
    )
    
    return result