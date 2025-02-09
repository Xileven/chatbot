import streamlit as st
from typing import List, Dict
import os
import requests
from llama_index import VectorStoreIndex, Document, ServiceContext
from llama_index.vector_stores import SimpleVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.embeddings import FastEmbedEmbedding
from llama_index.postprocessor import SentenceEmbeddingOptimizer

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
            # Initialize embedding model
            self.embed_model = FastEmbedEmbedding()
            
            # Create service context
            self.service_context = ServiceContext.from_defaults(
                embed_model=self.embed_model,
            )
            
            # Initialize vector store
            vector_store = SimpleVectorStore()
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Create some initial documents if needed
            documents = [
                Document(text="Welcome to the chatbot! This is an initial document to get started.")
            ]
            
            # Create vector store index
            self.index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                service_context=self.service_context,
                show_progress=True
            )
            
            # Initialize sentence embedding optimizer for reranking
            self.reranker = SentenceEmbeddingOptimizer(
                top_n=3,
                embed_model=self.embed_model
            )
            
            # Create query engine
            self.query_engine = self.index.as_query_engine(
                similarity_top_k=5,
                node_postprocessors=[self.reranker],
                response_mode="compact"
            )

        def query(self, prompt: str) -> Dict:
            try:
                response = self.query_engine.query(prompt)
                return {
                    'answer': str(response),
                    'sources': [f"Result {i+1}" for i in range(3)]  # Simplified source tracking
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

# App layout
st.set_page_config(page_title='ChatBot', layout='wide')

# Sidebar controls
with st.sidebar:
    st.header('Settings')
    st.session_state.web_search = st.toggle('Enable Web Search', value=True)
    st.session_state.show_citations = st.toggle('Show Citations', value=True)

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
