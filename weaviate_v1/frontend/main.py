import streamlit as st
import weaviate
from weaviate.classes.config  import Configure, Reconfigure
from weaviate.classes.query import MetadataQuery
import atexit
import base64
from PIL import Image
import io
# Page config
st.set_page_config(
    page_title="O-RAN RAG Assistant",
    page_icon="📡",
    layout="wide"
)

# Initialize connection (cached to avoid reconnecting every rerun)
@st.cache_resource
def init_weaviate():
    """Initialize Weaviate client"""
    try:
        client = weaviate.connect_to_local(
            host="172.17.0.4",
            port=8080,
            grpc_port=50051,
        )
        return client
    except Exception as e:
        st.error(f"Failed to connect to Weaviate: {e}")
        return None

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    
    # RAG parameters
    st.subheader("Retrieval Settings")
    num_results = st.slider("Number of chunks to retrieve", 1, 10, 3)
    
    st.subheader("Model Info")
    st.info("""
    **Embedding Model:** nomic-embed-text  
    **Generative Model:** llama3.2  
    **Database:** O-RAN Technical Specs
    """)
    
    # Clear chat button
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.conversation_history = []
        st.rerun()
    
    # Stats
    st.subheader("📊 Session Stats")
    st.metric("Messages", len(st.session_state.messages))

# Main interface
st.title("📡 O-RAN RAG Assistant")
st.markdown("Ask questions about O-RAN technical documentation")

# Initialize Weaviate
client = init_weaviate()

if client is None:
    st.error("Cannot connect to Weaviate. Please check your connection settings.")
    st.stop()


atexit.register(lambda: client.close())

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Show sources if available
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📚 View Sources"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**Source {i} (Page {source['page']}):**")
                    st.text(source['text'][:300] + "..." if len(source['text']) > 300 else source['text'])
                    st.divider()

# Chat input
if prompt := st.chat_input("Ask about O-RAN specifications..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching O-RAN documentation..."):
            try:
                collection = client.collections.get("Grounded_nomic")
                
                # Build context from conversation history
                history_context = ""
                if st.session_state.conversation_history:
                    recent_history = st.session_state.conversation_history[-2:]  # Last 2 turns
                    history_context = "Previous conversation context: " + "; ".join([
                        f"Q: {h['question']} A: {h['answer'][:100]}"
                        for h in recent_history
                    ])
                
                # Construct grouped task with history
                grouped_task = f"""Provided detailed answer to best of your ability, and cite source using trace. 
                                
                                Chat History: {history_context if history_context else ''}

                                If the answer cannot be found in the provided context, say so clearly."""
                
                # First, retrieve source chunks
                retrieval_response = collection.query.near_text(
                    query=prompt,
                    limit=num_results,
                    return_metadata=MetadataQuery(distance=True),
                    return_properties=["type", "page", "text", "description", "trace", "image"]
                )
                # Then generate answer with those chunks
                response = collection.generate.near_text(
                    query=prompt,
                    limit=num_results,
                    grouped_task=grouped_task
                )
                
                # Display answer
                answer = response.generated
                st.markdown(answer)
                
                # Collect sources
                sources = []
                for obj in retrieval_response.objects:
                    print( obj.properties)
                    sources.append({
                        "type": obj.properties.get("type", "Unkown"),
                        "page": obj.properties.get("page", "Unknown"),
                        "text": obj.properties.get("text", "Unknown"),
                        "description": obj.properties.get("description", "Unknown"),
                        "trace": obj.properties.get("trace", "Unknown"),
                        #"distance": obj.metadata.distance if obj.metadata else None,
                        "image": obj.properties.get("image")
                    })
                # Show sources in expander
                with st.expander("📚 View Sources"):
                    for i, source in enumerate(sources, 1):
                        st.markdown(f"**Source {i} (Page {source['page']}**")
                        st.text(f"Type: {source['type']}")
                        st.text(f"Trace: {source['trace']}")
                        st.text(source['text'][:300] + "..." if len(source['text']) > 300 else source['text'])
                        if source['description']:
                            st.caption(f"Description: {source['description']}")
                        image_b64 = source["image"]
                        print(f"Image data length: {len(image_b64) if image_b64 else 0}")
                        if image_b64:
                            try:
                                st.write("**Image:**")
                                # Decode base64 string to bytes
                                image_bytes = base64.b64decode(image_b64)
                                # Convert bytes to PIL Image
                                image = Image.open(io.BytesIO(image_bytes))
                                # Display image
                                st.image(image, width='content')
                            except Exception as e:
                                st.error(f"Error displaying image: {str(e)}")
                                st.write(f"Image data length: {len(image_b64) if image_b64 else 0}")
                        else:
                            st.info("📷 No image for this chunk")


                        st.divider()
                
                # Add to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
                
                # Update conversation history for context
                st.session_state.conversation_history.append({
                    "question": prompt,
                    "answer": answer
                })
                
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

# Footer
st.divider()
st.caption("Powered by Weaviate, Ollama, and Streamlit | Multimodal O-RAN RAG System")