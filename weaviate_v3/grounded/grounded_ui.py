import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from weaviate_v3.grounded.grounded_rag import ChatRAG
import base64
from PIL import Image
import io
import atexit

# Page config
st.set_page_config(
    page_title="O-RAN RAG Assistant",
    page_icon="📡",
    layout="wide"
)

# Initialize session state
if "chat_rag" not in st.session_state:
    st.session_state.chat_rag = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "initialized" not in st.session_state:
    st.session_state.initialized = False

# Sidebar - Configuration
with st.sidebar:
    st.title("⚙️ Settings")
    
    # Connection settings
    st.subheader("Weaviate Connection")
    weaviate_host = st.text_input("Host", "172.17.0.5")
    weaviate_port = st.number_input("Weaviate Port", min_value=1, max_value=65535, value=8080, step=1)
    
    st.divider()
    
    # Model settings
    st.subheader("Model Configuration")
    ollama_host = st.text_input(
        "Ollama Host",
        value="127.0.0.1",
        help="Where Ollama listens (e.g. 127.0.0.1 if `ollama serve` runs on this machine, or a Docker bridge IP like 172.17.0.4).",
    )
    ollama_model = st.text_input("Ollama Model", "llama3.2")
    ollama_port = st.number_input("Ollama Port", min_value=1, max_value=65535, value=11434, step=1)
    collection_name = st.text_input("Collection Name", "Grounded_nomic_full")
    multimodal = st.checkbox("Multimodal Mode (for vision models)", value=False)
    
    if multimodal:
        st.info("💡 Use models like: llama3.2-vision, llava, bakllava")
    
    st.divider()
    
    # RAG settings
    st.subheader("RAG Settings")
    use_rag = st.checkbox("Enable RAG", value=True)
    num_results = st.slider("Number of chunks to retrieve", 1, 10, 3)
    block_filter_label = st.selectbox("Block Filter", ["all", "chunks"], index=0)
    block_filter = None if block_filter_label == "all" else block_filter_label
    
    st.divider()
    
    # Initialize button
    if st.button("🚀 Initialize System", type="primary"):
        with st.spinner("Initializing ChatRAG system..."):
            try:
                st.session_state.chat_rag = ChatRAG(
                    collection_name=collection_name,
                    weaviate_host=weaviate_host,
                    weaviate_port=int(weaviate_port),
                    ollama_host=ollama_host,
                    ollama_model=ollama_model,
                    ollama_port=int(ollama_port),
                    default_block_filter=block_filter
                )
                st.session_state.initialized = True
                st.success("✅ System initialized!")
            except Exception as e:
                st.error(f"❌ Initialization failed: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
    
    # Clear chat button
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        if st.session_state.chat_rag:
            st.session_state.chat_rag.clear_history()
        st.rerun()
    
    # Stats
    st.divider()
    st.subheader("📊 Session Stats")
    st.metric("Messages", len(st.session_state.messages))

# Main interface
st.title("📡 O-RAN RAG Assistant")
st.markdown("Ask questions about O-RAN technical documentation")

# Show initialization status
if not st.session_state.initialized:
    st.info("👈 Configure settings in the sidebar and click 'Initialize System' to start chatting")
    
    # Quick start guide
    with st.expander("🚀 Quick Start Guide"):
        st.markdown("""
        ### Prerequisites
        1. **Weaviate** running
           ```bash
           docker run -d -p 8080:8080 -p 50051:50051 semitechnologies/weaviate:latest
           ```
        
        2. **Ollama** running with model pulled
           ```bash
           ollama pull llama3.2
           ollama pull nomic-embed-text
           ```
        
        ### Adding Documents
        Use the ChatRAG class to add documents:
        ```python
        from chat_rag import ChatRAG
        
        rag = ChatRAG(
            collection_name="Grounded_nomic",
            weaviate_host="172.17.0.6",
            weaviate_port=8080,
            ollama_model="llama3.2",
            ollama_port=11434
        )
        
        rag.add_documents([
            {
                "text": "Your document text...",
                "page": 1,
                "block_type": "text",
                "description": "Description",
                "trace": "doc_001",
                "images": None  # or base64 encoded image
            }
        ])
        ```
        """)
    
    st.stop()

# Register cleanup on app close
if st.session_state.chat_rag:
    atexit.register(lambda: st.session_state.chat_rag.close())

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show sources if available
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📚 View Sources"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**Source {i}**")
                    
                    # Show metadata
                    col1, col2, col3, col4,col5 = st.columns(5)
                    col1.text(f"Type: {source.get('block_type', source.get('type', 'Unknown'))}")
                    col2.text(f"Page: {source.get('page', 'Unknown')}")
                    col3.text(f"Trace: {source.get('trace', 'Unknown')}")
                    col4.text(f"Distance: {source.get('weaviate_distance', 'N/A'):.4f}" if source.get('weaviate_distance') else "Distance: N/A")
                    col5.text(f"File: {source.get('filename', 'Unknown')}")
                    # Show text content
                    text = source.get('text', '')
                    st.text(text[:300] + "..." if len(text) > 300 else text)
                    
                    # Show description if available
                    if source.get('description'):
                        st.caption(f"Description: {source['description']}")
                    
                    # Show image if available
                    image_b64 = source.get("images") or source.get("image")
                    if image_b64:
                        try:
                            st.write("**Image:**")
                            image_bytes = base64.b64decode(image_b64)
                            image = Image.open(io.BytesIO(image_bytes))
                            st.image(image, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error displaying image: {str(e)}")
                    else:
                        st.info("📷 No image for this chunk")
                    
                    st.divider()

# Chat input
if prompt := st.chat_input("Ask about O-RAN specifications..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Show thinking indicator
        with st.spinner("Searching O-RAN documentation..."):
            try:
                # Get response from ChatRAG
                result = st.session_state.chat_rag.chat(
                    user_message=prompt,
                    use_rag=use_rag,
                    top_k=num_results,
                    return_sources=True,
                    block_filter=block_filter
                )
                
                answer = result['answer']
                sources = result.get('sources', [])
                
                # Display answer
                message_placeholder.markdown(answer)
                
                # Show sources in expander
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(sources, 1):
                        st.markdown(f"**Source {i}**")
                        
                        # Show metadata
                        col1, col2, col3, col4,col5 = st.columns(5)
                        col1.text(f"Type: {source.get('block_type', source.get('type', 'Unknown'))}")
                        col2.text(f"Page: {source.get('page', 'Unknown')}")
                        col3.text(f"Trace: {source.get('trace', 'Unknown')}")
                        col4.text(f"Distance: {source.get('weaviate_distance', 'N/A'):.4f}" if source.get('weaviate_distance') else "Distance: N/A")
                        col5.text(f"File: {source.get('filename', 'Unknown')}")
                        # Show text content
                        text = source.get('text', '')
                        st.text(text[:300] + "..." if len(text) > 300 else text)
                        
                        # Show description if available
                        if source.get('description'):
                            st.caption(f"Description: {source['description']}")
                         
                        # Show image if available
                        image_b64 = source.get("images") or source.get("image")
                        if image_b64:
                            try:
                                st.write("**Image:**")
                                image_bytes = base64.b64decode(image_b64)
                                image = Image.open(io.BytesIO(image_bytes))
                                st.image(image, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error displaying image: {str(e)}")
                        else:
                            st.info("📷 No image for this chunk")
                        
                        st.divider()
                
                # Add to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                message_placeholder.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })
                import traceback
                st.error(traceback.format_exc())

# Footer
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption(f"💬 Messages: {len(st.session_state.messages)}")
with col2:
    st.caption(f"🔍 RAG: {'Enabled' if use_rag else 'Disabled'}")
with col3:
    st.caption(f"🤖 Model: {ollama_model if st.session_state.initialized else 'Not initialized'}")
