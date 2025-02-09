import streamlit as st
from typing import List, Dict
import os
import requests
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
from llama_index.embeddings.fastembed import FastEmbedEmbedding

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
                uri=os.getenv('ZILLIZ_URI'),  # Should be "https://xxx.api.region.zillizcloud.com"
                token=os.getenv('ZILLIZ_TOKEN'),
                secure=True
            )
            
            from pymilvus import FieldSchema, CollectionSchema, DataType

            # Create collection if not exists
            if not connections.has_collection("chatbot_data"):
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
            self.embed_model = FastEmbedEmbedding()

        def query(self, prompt: str) -> Dict:
            query_embedding = self.embed_model.get_text_embedding(prompt)
            search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=3,
                output_fields=["content"]
            )
            return {
                'answer': '\n'.join([hit.entity.get('content') for hit in results[0]]),
                'sources': [f"VectorDB result {i+1}" for i in range(len(results[0]))]
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
