import streamlit as st
st.config.set_option('server.fileWatcherType', 'none')

import nest_asyncio
nest_asyncio.apply()

import os
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
import google.generativeai as genai

from llama_index.llms.deepseek import DeepSeek

from llama_index.core import VectorStoreIndex, Settings, Document
from llama_index.core.node_parser import SentenceSplitter, MarkdownElementNodeParser
from llama_index.core.schema import TextNode
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_parse import LlamaParse
import requests 
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import Document
from llama_index.readers.file import (
    DocxReader,
    PandasExcelReader,
)

# Set page config
st.set_page_config(page_title="Domain Knowledge Augmented LLM Demo", layout="wide")
st.title("Domain Knowledge Augmented LLM")
st.subheader("Demo, BAMA, Feb 2025")

st.write("""
##### This demo is a POC of:
    1. LLM interacts with user specified documents.
    2. 3 files are ingested at the same time
         - PDF, Schwab 2024 Q10
         - Word, Schwab 2023 Q10 (Converted from PDF to Word)
         - Excel, A table extracted from Schwab 2022 K10 (page 26)
         
##### Due to limitation of hardware (memory, storage, GPU, API), demo is restricted from
    1. Reasoning(Ambiguous questions)
    2. Sematic questioning (follow up questions)
    3. Large model (accuracy, tradeoff between speed and accuracy)
    4. Fine-tune (optimization)
    5. Unstable output
         


* [SEC 10K 2022](https://content.schwab.com/web/retail/public/about-schwab/SEC_Form10k_2022.pdf)
* [SEC 10Q 2023](https://content.schwab.com/web/retail/public/about-schwab/SEC_Form10-Q_093023.pdf)
* [SEC 10Q 2024](https://content.schwab.com/web/retail/public/about-schwab/SEC_Form10Q_093024.pdf)
         
""")

import dotenv
dotenv.load_dotenv()

os.environ["LLAMA_CLOUD_API_KEY"] = os.getenv("LLAMA_CLOUD_API_KEY")
os.environ["MILVUS_API_KEY"] = os.getenv("ZILLIZ_API_KEY")
MILVUS_API_KEY = os.getenv('ZILLIZ_API_KEY')

# Using OpenAI API for embeddings/llms
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")  
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")  # from example

os.environ["DEEPSEEK_API_KEY"] = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')

# Tavily API key
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')


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
            topic="news",
            time_range="w",
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
# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Add web search toggle
st.sidebar.title("Search Options")
use_web_search = st.sidebar.toggle("Enable Web Search", value=False)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know about Schwab?"):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response
    with st.chat_message("assistant"):
        if use_web_search:
            response_placeholder = st.empty()
            response_placeholder.markdown("🤔 Searching both local documents and the web...")
            
            # Get RAG results with citations
            rag_response = recursive_query_engine.query(prompt)
            rag_response_text = str(rag_response)
            
            # Get web search results
            web_response = web_search(prompt)
            
            # Format the combined response with citations
            combined_response = f"""
### Local Document Search Results:
{rag_response_text}

### Web Search Results:
{web_response}
"""
            response_placeholder.markdown(combined_response)
            st.session_state.messages.append({"role": "assistant", "content": combined_response})
        else:
            response_placeholder = st.empty()
            response_placeholder.markdown("🤔 Searching local documents...")
            
            # Get RAG results with citations
            response = recursive_query_engine.query(prompt)
            response_text = str(response)
            
            response_placeholder.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})

# Display sample questions
st.markdown("""
**Sample Questions:**
```
- [PDF] How Integration of Ameritrade impact client metrics from 2023 to 2024?
- [Excel] Where is the headquarters of schwab and what is its size, including leased and owned
- [PDF & Word] Compare Client Metrics of Three Month Ended from 2022, to 2023, to 2024, in numbers, and printout in table
- [PDF & Word] Compare Total Net Revenue from 2022, to 2023, to 2024 and printout in table
- [Summary] based on Client Metrics of Three Month Ended from 2022, to 2023, to 2024, analyze the business
- Compare Total Net Revenue from 2022, to 2023, to 2024 and printout in table
```
""")
