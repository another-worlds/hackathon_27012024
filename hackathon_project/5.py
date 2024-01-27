from operator import itemgetter

from langchain_community.vectorstores.faiss import FAISS
#from langchain_community.vectorstores.pinecone import Pinecone
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.output_parsers import JsonOutputParser
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, create_openai_tools_agent

from langchain.globals import set_verbose, set_debug


from langchain.schema import format_document
from langchain_core.messages import AIMessage, HumanMessage, get_buffer_string
from langchain_core.runnables import RunnableParallel
from langchain.prompts.prompt import PromptTemplate

from operator import itemgetter

from langchain.memory import ConversationBufferMemory
from tool_helper import get_price_change_tool

from dotenv import load_dotenv
load_dotenv()

#retriever_chain = RetrievalQA()

# load_dotenv()
# def update_db(db: FAISS, model :ChatOpenAI, query) -> FAISS:
#     model.bind_functions(get_price_change_tool)
# vectorstore = FAISS.from_texts(
#     [""], embedding=OpenAIEmbeddings()
# )

import pinecone
import os
import os
from pinecone import Pinecone, ServerlessSpec
pc = Pinecone(
    api_key=os.environ["PINECONE_API_KEY"]
)

index_name = 'rag'
existing_indexes = [
    index_info["name"] for index_info in pc.list_indexes()
]

if 'rag' not in pc.list_indexes().names():
    pc.create_index(
        name='rag',
        dimension=1536,
        metric='euclidean',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-west-2'
        )
    )
index = pc.Index(index_name)


embed_model = OpenAIEmbeddings(model="text-embedding-ada-002")


from langchain.vectorstores.pinecone import Pinecone

text_field = "text"  # the metadata field that contains our text

# initialize the vector store object
vectorstore = Pinecone(
    index, embed_model.embed_query, text_field
)
retriever = vectorstore.as_retriever()


model = ChatOpenAI()



import pandas as pd
from langchain.tools.render import render_text_description
# tds = pd.read_csv('tickers.csv', index_col=0)
# tds = '\n'.join([f"{r.key} - {r.value}" for r in tds.to_dict()['T']])
ds = pd.read_csv('tickers.csv', index_col=0)

retriever_tool = create_retriever_tool(
    retriever,
    "search_ticker_data",
    "Searches and returns the ticker data. Mandatory to use",
)
tools = [retriever_tool, get_price_change_tool]
rendered_tools = render_text_description(tools)


messages = []
prompt0 = 'This is the context: {context}'
system_prompt = f"""

The previous messages were the context. You are an assistant designed to get 
ticker prices per timeframe, If the context is the same by the number of days and the ticker,
answer using the context.
ticker information: {str(ds.head(100))}
example of formatting:
bitcoin -> btc
solana -> sol
ONLY IF THE CONTEXT DOESN'T INCLUDE RELEVANT NUMBER OF DAYS AND ASSET NAME, 
return the name and input of the following tool you will use. Return your response as a JSON blob with 'name' and 'arguments' keys.
{render_text_description([get_price_change_tool])},

"""

system_prompt2 = "Using the dictionary provided, answer to to the user query."

prompt = ChatPromptTemplate.from_messages(
    [("system", prompt0),
     ("system", system_prompt), 
     ("user", "{input}")
    ]
)

# agent = create_openai_tools_agent(model, tools, prompt)
# agent_executor = AgentExecutor(agent=agent, tools=tools)
def tool_chain(model_output):
    tool_map = {tool.name: tool for tool in tools}
    chosen_tool = tool_map[model_output["name"]]
    return itemgetter("arguments") | chosen_tool
#set_debug(True)


args_chain = (
    prompt | model | JsonOutputParser() | RunnablePassthrough.assign(output=tool_chain)# | prompt2
)


#retrieval_chain = RetrievalQA.from_llm(model,prompt=args_chain,retriever=retriever)

while True:
    #query = input()
    query = "What was the price of bitcoin 2 days ago"
    similar = vectorstore.similarity_search(query, k=30)
    pages = '\n'.join([q.page_content for q in similar])
    context = f"Context: {pages}"
    clown = "-----------------"
    print(f"{clown}CONTEXT{clown}\n: {context}")
    #print(similar)
    # similar
    result = args_chain.invoke({"input": query, "context": vectorstore.similarity_search(query, k=3)})
    print(f"{clown}RESULT{clown}\n: {result}")
    #print(f"{clown}RESULT DICT{result.dict()}{clown}\n")
    arguments = str(result['arguments'])
    output = str(result['output'])
    vectorstore._embed_query(f"arguments: "+ arguments + " output: " + output )
    
    print(arguments)
    print(output)
    break
    #entry = text, metadata
    #vectorstore.add_texts([result], metadatas=[])
    # embed_model.embed_query(query)
    # embed_model.update_forward_refs()
