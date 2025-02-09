import streamlit as st
from typing import List, Dict
import os
import requests
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker

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
            # Validate environment variables first
            if not os.getenv('ZILLIZ_URI') or not os.getenv('ZILLIZ_TOKEN'):
                raise ValueError("Missing Zilliz credentials in environment variables")
            
            # Connect using Zilliz Cloud format
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
                dim=768  # FastEmbed dimension
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
                
                # Create index
                index_params = {
                    "index_type": "IVF_FLAT",
                    "metric_type": "L2",
                    "params": {"nlist": 128}
                }
                self.collection.create_index("embedding", index_params)
            else:
                self.collection = Collection("chatbot_data")
            
            self.collection.load()
            
            # Initialize embedding model
            self.embed_model = FastEmbedEmbedding()
            
            # Create vector store index
            self.index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                show_progress=True
            )
            
            # Initialize reranker
            self.reranker = FlagEmbeddingReranker(
                model="BAAI/bge-reranker-large",
                top_n=3
            )
            
            # Create query engine
            self.query_engine = self.index.as_query_engine(
                similarity_top_k=5,
                node_postprocessors=[self.reranker],
                verbose=True,
                synthesize=True
            )

        def query(self, prompt: str) -> Dict:
            try:
                response = self.query_engine.query(prompt)
                return {
                    'answer': str(response),
                    'sources': [f"VectorDB result {i+1}" for i in range(3)]  # Simplified source tracking
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

def validate_zilliz_connection():
    try:
        connections.connect(
            alias="default",
            uri=os.getenv('ZILLIZ_URI'),
            token=os.getenv('ZILLIZ_TOKEN'), 
            secure=True
        )
        return True, "Connection successful"
    except Exception as e:
        return False, f"Connection failed: {str(e)}"

# App layout
st.set_page_config(page_title='ChatBot', layout='wide')

# Sidebar controls
with st.sidebar:
    st.header('Settings')
    st.session_state.web_search = st.toggle('Enable Web Search', value=True)
    st.session_state.show_citations = st.toggle('Show Citations', value=True)
    if st.button("Test Zilliz Connection"):
        success, message = validate_zilliz_connection()
        if success:
            st.success(message)
        else:
            st.error(message)
            st.markdown("""
                **Required format:**  
                `ZILLIZ_URI=https://[cluster-id].api.[region].zillizcloud.com`  
                `ZILLIZ_TOKEN=your_db_token`
            """)

# Chat interface
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
