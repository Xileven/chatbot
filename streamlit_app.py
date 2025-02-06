import streamlit as st
import nest_asyncio
nest_asyncio.apply()

import os
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
# import google.generativeai as genai

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
         - Word, Schwab 2023 Q10
         - Excel, A table extracted from Schwab 2022 K10
         
##### Due to limitation of hardware (memory, storage, API), demo is restricted from
    1. Reasoning(Ambiguous questions)
    2. Sematic questioning (follow up questions)
    3. Large model (multiple languages and more accuracy)
    4. Model tuning (optimization)
    5. Unstable output
         


* [SEC 10K 2022](https://content.schwab.com/web/retail/public/about-schwab/SEC_Form10k_2022.pdf)
* [SEC 10Q 2023](https://content.schwab.com/web/retail/public/about-schwab/SEC_Form10-Q_093023.pdf)
* [SEC 10Q 2024](https://content.schwab.com/web/retail/public/about-schwab/SEC_Form10Q_093024.pdf)
         
""")

import dotenv
dotenv.load_dotenv()

# API access to llama-cloud
os.environ["LLAMA_CLOUD_API_KEY"] = os.getenv("LLAMA_CLOUD_API_KEY")
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

# Using OpenAI API for embeddings/llms
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")  
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")  # from example

os.environ["DEEPSEEK_API_KEY"] = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')


# Model Selection UI
embed_model_option = st.selectbox(
    "Select Embedding Model",
    ["OpenAI text-embedding-3-small", "Google text-embedding-004"]
)

llm_model_option = st.selectbox(
    "Select LLM Model",
    ["OpenAI GPT-3.5", "DeepSeek Reasoner", "Gemini 2.0"]
)

# Global Settings
@st.cache_resource
def initialize_models():
    # Initialize embedding model based on selection
    if embed_model_option == "OpenAI text-embedding-3-small":
        embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    else:
        embed_model = GeminiEmbedding(
            model_name='text-embedding-004', 
            api_key=GOOGLE_API_KEY
        )

    # Initialize LLM based on selection
    if llm_model_option == "OpenAI GPT-3.5":
        llm = OpenAI(model="gpt-3.5-turbo-0125")
    elif llm_model_option == "DeepSeek Reasoner":
        llm = DeepSeek(
            model="deepseek-reasoner", 
            api_key=DEEPSEEK_API_KEY
        )
    else:
        llm = Gemini(
            model="models/gemini-2.0-flash-001",
            api_key=GOOGLE_API_KEY
        )

    Settings.llm = llm
    Settings.embed_model = embed_model
    return llm

llm = initialize_models()

# Initialize parsers
@st.cache_resource
def initialize_parsers():
    return LlamaParse(result_type="markdown")

llama_parser = initialize_parsers()

# Initialize document readers
@st.cache_resource
def initialize_readers():
    llama_parser = LlamaParse(result_type="markdown")    
    docx_reader = DocxReader()
    
    file_extractor = {
        # Document formats
        ".pdf": llama_parser,  # Converts PDF to markdown
        ".docx": docx_reader,
        ".doc": docx_reader,
        ".txt": None,  # Default reader for text files
        ".xlsx": PandasExcelReader(),  # Excel files (newer format)
        ".xls": PandasExcelReader(),   # Excel files (older format)
    }
    return file_extractor

file_extractor = initialize_readers()

import glob
print(f"Files in './FILES': {glob.glob('./FILES/*')}")

# Add a button to load and process documents
if st.button("[click] Load and Process Documents (about 5 min)"):
    with st.spinner("Loading and processing documents..."):
        try:
            # Load documents from FILES directory
            documents = SimpleDirectoryReader(
                "./FILES", 
                file_extractor=file_extractor,
                filename_as_id=True  
            ).load_data()
            
            if documents:
                st.session_state.documents = documents
                st.success(f"Successfully loaded {len(documents)} documents from FILES directory")
            else:
                st.warning("No documents found in FILES directory")
        except Exception as e:
            st.error(f"Error loading documents: {str(e)}")

# Process documents and create index
if 'documents' in st.session_state:
    if 'index_created' not in st.session_state:
        with st.spinner("Processing documents and creating index..."):
            # Create nodes using both approaches
            text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=100)
            page_nodes = []
            for doc in st.session_state.documents:
                text_chunks = text_splitter.split_text(doc.text)
                for i, chunk in enumerate(text_chunks):
                    node = TextNode(
                        text=chunk,
                        metadata={
                            "file_name": doc.metadata.get("file_name", ""),
                            "chunk_index": i,
                        }
                    )
                    page_nodes.append(node)

            # Parse markdown structure
            node_parser = MarkdownElementNodeParser(llm=llm, num_workers=8)
            nodes = node_parser.get_nodes_from_documents(st.session_state.documents)
            base_nodes, objects = node_parser.get_nodes_and_objects(nodes)

            # Combine all nodes
            all_nodes = base_nodes + objects + page_nodes

            # Create index
            recursive_index = VectorStoreIndex(all_nodes, show_progress=True)
            
            # Create query engine
            reranker = FlagEmbeddingReranker(
                model="BAAI/bge-reranker-large",
                top_n=3,
            )
            
            recursive_query_engine = recursive_index.as_query_engine(
                similarity_top_k=5,
                node_postprocessors=[reranker],
                verbose=True,
                synthesize=True
            )
            
            st.session_state.query_engine = recursive_query_engine
            st.session_state.index_created = True
            st.success("Index created successfully!")

    # Create the chat interface
    st.header("Chat with your Documents")

    # Display sample questions
    st.markdown("""
    **Sample Questions:**
    ```
    - [PDF] How Integration of Ameritrade impact client metrics from 2023 to 2024?
    - [Excel] Where is the headquarters of schwab and what is its size, including leased and owned
    - [PDF & Word] Compare Total Net Revenue from 2022, to 2023, to 2024 and printout in table
    - Compare Client Metrics from 2023 to 2024, and from 2022 to 2023, in numbers
    - Compare Client Metrics of Three Month Ended from 2022, to 2023, to 2024, in numbers, and printout in table
    - Compare Total Net Revenue from 2022, to 2023, to 2024 and printout in table
    ```
    """)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask a question about your documents"):
        # Display user message in chat message container
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.query_engine.query(prompt)
                st.markdown(response.response)
                st.session_state.messages.append({"role": "assistant", "content": response.response})

else:
    st.info("Please load the documents first by clicking the 'Load and Process Documents' button above.")
