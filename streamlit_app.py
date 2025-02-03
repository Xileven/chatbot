import streamlit as st
import nest_asyncio
nest_asyncio.apply()

import os
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
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
)

# Set page config
st.set_page_config(page_title="Domain Knowledge Augmented LLM Demo", layout="wide")
st.title("Domain Knowledge Augmented LLM Demo")
st.subheader("BAMA, Feb 2025")
st.write("""
This demo is a POC of using Large Language Model(LLM) to interact with user specified documents.

Data used in this demo is 2023 Q-10 and 2024 Q-10 of Schwab from public Schwab website.

* [https://content.schwab.com/web/retail/public/about-schwab/SEC_Form10-Q_093023.pdf](https://content.schwab.com/web/retail/public/about-schwab/SEC_Form10-Q_093023.pdf)
* [https://content.schwab.com/web/retail/public/about-schwab/SEC_Form10Q_093024.pdf](https://content.schwab.com/web/retail/public/about-schwab/SEC_Form10Q_093024.pdf)
""")

import dotenv
dotenv.load_dotenv()

# Set API keys from Streamlit secrets
os.environ["LLAMA_CLOUD_API_KEY"] = os.getenv("LLAMA_CLOUD_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")

# Global Settings
@st.cache_resource
def initialize_models():
    embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    llm = OpenAI(model="gpt-3.5-turbo-0125")
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
        ".txt": None  # Default reader for text files
    }
    return file_extractor

file_extractor = initialize_readers()

# Add a button to load and process documents
if st.button("Load and Process Documents (about 5 min)"):
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
    - Compare Client Metrics from 2023 to 2024, and from 2022 to 2023, in numbers
    - Compare Client Metrics of Three Month Ended from 2022, to 2023, to 2024, in numbers, and printout in table
    - Compare Total Net Revenue from 2022, to 2023, to 2024 and printout in table
    - Compare Net Revenue in each category from 2022, to 2023, to 2024 and printout in table
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
