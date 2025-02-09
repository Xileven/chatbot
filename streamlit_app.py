import streamlit as st
from typing import List, Dict
import os
import requests
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
import pandas as pd

# Initialize session states
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'web_search' not in st.session_state:
    st.session_state.web_search = False
if 'show_citations' not in st.session_state:
    st.session_state.show_citations = True

# Initialize components
@st.cache_resource
def get_rag_system():
    class RagSystem:
        def __init__(self):
            # Validate Zilliz credentials
            if not os.getenv('ZILLIZ_URI') or not os.getenv('ZILLIZ_TOKEN'):
                raise ValueError("Missing Zilliz credentials in environment variables")
            
            # Connect to Milvus
            connections.connect(
                alias="default",
                uri=os.getenv('ZILLIZ_URI'),
                token=os.getenv('ZILLIZ_TOKEN'),
                secure=True
            )

            # Initialize vector store
            self.vector_store = MilvusVectorStore(
                uri=os.getenv('ZILLIZ_URI'),
                token=os.getenv('ZILLIZ_TOKEN'),
                collection_name="chatbot_data",
                dim=768
            )

            # Create or get collection
            if not utility.has_collection("chatbot_data"):
                fields = [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=2000),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)
                ]
                schema = CollectionSchema(fields, description="Chatbot knowledge base")
                self.collection = Collection("chatbot_data", schema)
                self.collection.create_index(
                    "embedding", 
                    {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}}
                )
            else:
                self.collection = Collection("chatbot_data")
            
            self.collection.load()
            
            # Initialize embedding model
            self.embed_model = FastEmbedEmbedding()
            
            # Create index with reranker
            self.index = VectorStoreIndex.from_vector_store(
                self.vector_store,
                show_progress=True
            )
            
            self.reranker = FlagEmbeddingReranker(
                model="BAAI/bge-reranker-large",
                top_n=3
            )
            
            self.query_engine = self.index.as_query_engine(
                similarity_top_k=5,
                node_postprocessors=[self.reranker],
                verbose=True
            )

        def query(self, prompt: str) -> Dict:
            try:
                response = self.query_engine.query(prompt)
                return {
                    'answer': str(response),
                    'sources': [f"VectorDB result {i+1}" for i in range(3)]
                }
            except Exception as e:
                return {
                    'answer': f"Error occurred: {str(e)}",
                    'sources': []
                }

    return RagSystem()

@st.cache_resource
def get_searcher():
    class WebSearcher:
        def search(self, query: str) -> List[str]:
            try:
                response = requests.get(
                    "https://serpapi.com/search",
                    params={
                        "q": query,
                        "api_key": os.getenv('SEARCH_API_KEY')
                    }
                )
                results = response.json().get('organic_results', [])[:3]
                return [f"{res['title']} ({res['link']})" for res in results]
            except Exception as e:
                return [f"Search error: {str(e)}"]

    return WebSearcher()

# App layout (kept identical to original)
st.set_page_config(page_title='ChatBot', layout='wide')

# Sidebar controls (identical UI)
with st.sidebar:
    st.header('Settings')
    st.session_state.web_search = st.toggle('Enable Web Search', value=True)
    st.session_state.show_citations = st.toggle('Show Citations', value=True)
    if st.button("Test Zilliz Connection"):
        try:
            connections.connect(
                alias="default",
                uri=os.getenv('ZILLIZ_URI'),
                token=os.getenv('ZILLIZ_TOKEN'),
                secure=True
            )
            st.success("Connection successful")
        except Exception as e:
            st.error(f"Connection failed: {str(e)}")

# Chat interface (identical UI)
for message in st.session_state.messages:
    with st.chat_message(message['role'], avatar=message.get('avatar')):
        st.markdown(message['content'])
        if st.session_state.show_citations and message.get('citations'):
            st.caption('Sources:')
            for cit in message['citations']:
                st.markdown(f'- {cit}')

if prompt := st.chat_input('Ask me anything...'):
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    
    with st.chat_message('user', avatar='👤'):
        st.markdown(prompt)

    with st.chat_message('assistant', avatar='🤖'):
        response = ''
        citations = []
        
        if st.session_state.web_search:
            search_results = get_searcher().search(prompt)
            response += '**Web Search Results:**\n' + '\n'.join(search_results[:3])
            citations.extend(search_results)
        else:
            rag_response = get_rag_system().query(prompt)
            response = rag_response['answer']
            citations = rag_response.get('sources', [])

        st.markdown(response)
        if citations and st.session_state.show_citations:
            st.caption('Sources:')
            for cit in citations[:3]:
                st.markdown(f'- {cit}')

    st.session_state.messages.append({
        'role': 'assistant',
        'content': response,
        'citations': citations
    })