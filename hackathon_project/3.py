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

index_name = 'bitcoin-price1'
existing_indexes = [
    index_info["name"] for index_info in pc.list_indexes()
]

if 'bitcointest' not in pc.list_indexes().names():
    pc.create_index(
        name='bitcointest',
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
embed_model.embed_query("Bitcoin was 45000 2 days ago")

model = ChatOpenAI()

# Create RAG entry
_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)


# Default context for answering question
template = """Answer the about asset price based only on the following context:
{context}

If the relevant context isn't provided, use this tool: 

Question: {question}
"""
ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

# Create retrieval prompt
DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template="{page_content}")

def _combine_documents(
    docs, 
    document_prompt=DEFAULT_DOCUMENT_PROMPT, 
    document_separator="\n\n"
    ):
    
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


# First we add a step to load memory
# This adds a "memory" key to the input object

# -------------------------------
# loaded_memory = RunnablePassthrough.assign(
#     chat_history=RunnableLambda() | itemgetter("history"),
# )

# # Now we calculate the standalone question
# standalone_question = {
#     "standalone_question": {
#         "question": lambda x: x["question"],
#         "chat_history": lambda x: get_buffer_string(x["chat_history"]),
#     }
#     | CONDENSE_QUESTION_PROMPT
#     | ChatOpenAI(temperature=0)
#     | StrOutputParser(),
# }
# def tool_chain(model_output):
#     return itemgetter("arguments") | get_price_change_tool

# -------------------------------
#render tool desctiption

import pandas as pd
from langchain.tools.render import render_text_description
# tds = pd.read_csv('tickers.csv', index_col=0)
# tds = '\n'.join([f"{r.key} - {r.value}" for r in tds.to_dict()['T']])
ds = pd.read_csv('tickers.csv', index_col=0)

retriever_tool = create_retriever_tool(
    retriever,
    "search_ticker_data",
    "Searches and returns the ticker data. Mandatory to us",
)
tools = [retriever_tool, get_price_change_tool]
rendered_tools = render_text_description(tools)

messages = []
system_prompt1 = f"""You are an assistant, who converts data into specific format
You need to output query in the following manner:
"Price of 'ticker' 'days' ago" where ticker and days are provided in the prompt.
Week means 7 days, month means 30 days.
for the ticker substitution look at this table: {str(ds.head(100))} example of formatting
bitcoin -> btc
solana -> sol
""" 

system_prompt2 = f"""You are an assistant, who uses the function: 
"{render_text_description([retriever_tool])}" to get relevant information according
to the requested data.

If there was relevant information with specific "days" and "ticker" mentions,
return this information.

ticker database: {str(ds.head(100))}

If the relevant information wasn't found

"""

system_prompt3 = f"""You are an assistant that has access to the following set of tools. Here are the names and descriptions for each tool:
{render_text_description([get_price_change_tool])}

ticker information: {str(ds.head(100))}
example of formatting:
bitcoin -> btc
solana -> sol

Given the user input, return the name and input of the tool to use. Return your response as a JSON blob with 'name' and 'arguments' keys."""

print(rendered_tools)
prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt1),
     ("system", system_prompt2),
     ("system", system_prompt3), 
     ("user", "{input}")]
)

# agent = create_openai_tools_agent(model, tools, prompt)
# agent_executor = AgentExecutor(agent=agent, tools=tools)
def tool_chain(model_output):
    tool_map = {tool.name: tool for tool in tools}
    chosen_tool = tool_map[model_output["name"]]
    return itemgetter("arguments") | chosen_tool


args_chain = (
    prompt | model | JsonOutputParser() | RunnablePassthrough.assign(output=tool_chain)
)
# while True:
#     query = input()
#     print(args_chain.invoke({"input": query}))


# -------------------------------

# Now we retrieve the documents
retrieved_documents = {
    "docs": itemgetter("standalone_question") | retriever,
    "question": lambda x: x["standalone_question"],
}
# Now we construct the inputs for the final prompt
final_inputs = {
    "context": lambda x: _combine_documents(x["docs"]),
    "question": itemgetter("question"),
}
# And finally, we do the part that returns the answers
answer = {
    "answer": final_inputs | ANSWER_PROMPT | ChatOpenAI(),
    "docs": itemgetter("docs"),
}
# And now we put it all together!
final_chain = loaded_memory | standalone_question | retrieved_documents | answer

inputs = {"question": "How much did bitcoin cost 2 days ago"}
result = final_chain.invoke(inputs)
memory.save_context(inputs, {"answer": result["answer"].content})
memory.load_memory_variables({})
print(result)

inputs = {"question": "What was the price of bitcoin 2 days ago"}
result = final_chain.invoke(inputs)
result

print(result)