

# from https://github.com/run-llama/llama_parse/blob/main/examples/demo_advanced.ipynb

#%%
# ====================================================================================================
# ====================================================================================================
# llama-parse is async-first, running the async code in a notebook requires the use of nest_asyncio
print ("asyncio")
import nest_asyncio
nest_asyncio.apply()


print("dotenv")
# API access to llama-cloud
import dotenv
dotenv.load_dotenv('/Users/jinwenliu/github/.env/.env')

# # Reload environment to ensure we have the latest values
# dotenv.load_dotenv('/Users/jinwenliu/github/.env/.env', override=True)


#%%
# ====================================================================================================
print ("load API keys")
# ====================================================================================================
import os
# API access to llama-cloud
# os.environ["LLAMA_CLOUD_API_KEY"] = os.getenv("LLAMA_CLOUD_API_KEY")
os.environ["MILVUS_API_KEY"] = os.getenv("ZILLIZ_API_KEY")
MILVUS_API_KEY = os.getenv('ZILLIZ_API_KEY')

# Using OpenAI API for embeddings/llms
# os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
# os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")  
# GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")  # from example

# os.environ["DEEPSEEK_API_KEY"] = os.getenv("DEEPSEEK_API_KEY")
# DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')

# Tavily API key
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')

#%%
# ====================================================================================================
print ("import")
# ====================================================================================================

import os

from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.embeddings.openai import OpenAIEmbedding

from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import StorageContext
from llama_index.core import Document
from llama_index.readers.file import (
    DocxReader,
    PandasExcelReader,
)
import pandas
from tavily import TavilyClient

# Import web search related packages
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.query_engine import RouterQueryEngine
import requests

import streamlit as st

#%%
# Function to perform web search using Tavily API directly
def web_search(query: str) -> str:
    if not TAVILY_API_KEY:
        return "Error: Tavily API key is not set. Please set the TAVILY_API_KEY environment variable."
    
    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)
        result = client.search(
            query=query,
            search_depth="advanced",
            # topic="news", # news will make it irrelevant, dont use it
            time_range="y",
            include_answer="advanced",
            max_results=5,
        )
        
        # Extract the answer and search results
        answer = result.get('answer', '')
        search_results = result.get('results', [])
        
        # Combine the information
        combined_info = [answer]
        for res in search_results:
            combined_info.append(f"- {res.get('title')}: {res.get('content')}")
        
        return "\n".join(combined_info)
    except Exception as e:
        return f"Error: {str(e)}"

#%%
# Function to combine RAG and web search results
def hybrid_search(query):
    # Get RAG results
    rag_response = recursive_query_engine.query(query)
    
    # Get web search results
    web_response = web_search(query)
    
    # Create tools for the final agent
    rag_tool = QueryEngineTool(
        query_engine=recursive_query_engine,
        metadata=ToolMetadata(
            name="rag_knowledge",
            description="Provides information from the local knowledge base"
        )
    )
    
    # Create the final agent to combine results
    final_agent = OpenAIAgent.from_tools(
        [rag_tool],
        verbose=True
    )
    
    # Combine the results
    combined_prompt = f"""
    Please provide a comprehensive answer based on both local knowledge and web search results:
    
    Local Knowledge: {rag_response}
    Web Search Results: {web_response}
    
    Synthesize both sources to provide the most up-to-date and accurate information.
    If the information from different sources conflicts, prefer more recent sources and explain the discrepancy.
    """
    
    final_response = final_agent.chat(combined_prompt)
    return final_response

#%%
# ====================================================================================================
print ("load Index Ready nodes from Milvus")
# ====================================================================================================
#%%
vector_store = MilvusVectorStore(
    uri="https://in03-421d8d9c7f4c34b.serverless.gcp-us-west1.cloud.zilliz.com",
    token=MILVUS_API_KEY,
    collection_name="bama_llm_demo__EMBED_text_embedding_ada_002__LLM_gpt_3P5_turbo_0125",
    dim=1536,  # 1536 is default dim for OpenAI

)

storage_context = StorageContext.from_defaults(vector_store=vector_store)

recursive_index = VectorStoreIndex.from_vector_store(
            vector_store = vector_store,
            # storage_context = storage_context,
            show_progress=True
            )


#%%
from llama_index.postprocessor.flag_embedding_reranker import (
    # pruning away irrelevant nodes from the context
    FlagEmbeddingReranker,
)
reranker = FlagEmbeddingReranker(
                                model="BAAI/bge-reranker-large",
                                top_n=5,
)

recursive_query_engine = recursive_index.as_query_engine(
                                        similarity_top_k=10, 
                                        node_postprocessors=[reranker], 
                                        verbose=True,
                                        synthesize=True
)



# %%
# Example query using hybrid search
# query = "Compare Client Metrics of Three Month Ended from 2022, to 2023, to 2024, in numbers, and printout in table"
query = "Summarize how Scharle Schwab Bank doing in 2024"
rag_response = recursive_query_engine.query(query)
print("====================================RAG Response====================================")
print(rag_response)

print("====================================Hybrid Response====================================")
hybrid_response = hybrid_search(query)
print(hybrid_response)


# %%
def main():
    # Initialize session state for chat history if it doesn't exist
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Set up the Streamlit page
    st.set_page_config(page_title="Hybrid Search Chatbot", layout="wide")
    st.title("Hybrid Search Chatbot")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = hybrid_search(prompt)
                st.markdown(response)
                
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": str(response)})

    # Add a sidebar with information
    with st.sidebar:
        st.title("About")
        st.markdown("""
        This chatbot combines:
        - RAG (Retrieval-Augmented Generation)
        - Web search capabilities
        - Local knowledge base
        
        It provides comprehensive answers by synthesizing information from multiple sources.
        """)

if __name__ == "__main__":
    main()
