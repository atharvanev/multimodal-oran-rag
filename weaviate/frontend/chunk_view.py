import streamlit as st
import weaviate
import base64
from PIL import Image
import io

st.title("Weaviate Data Viewer")

# Connect to remote server
client = weaviate.connect_to_local(
    host="172.17.0.4",  
    port=8080,
    grpc_port=50051,
)

questions = client.collections.get("Grounded_nomic")

# Get total count
total = questions.aggregate.over_all(total_count=True)
st.metric("Total Chunks", total.total_count)

# Pagination
page_size = st.slider("Items per page", 10, 100, 20)
offset = st.number_input("Offset", min_value=0, value=0, step=page_size)

# Fetch objects
response = questions.query.fetch_objects(
    limit=page_size,
    offset=offset,
    return_properties=["type", "page", "text", "description", "trace", "image"]
)

# Display
for i, obj in enumerate(response.objects, 1):
    with st.expander(f"{i}. Page {obj.properties.get('page')} - {obj.properties.get('type')}"):
        st.write(f"**UUID:** {obj.uuid}")
        st.write(f"**Type:** {obj.properties.get('type', 'TEXT')}")
        st.write(f"**Page:** {obj.properties.get('page', 'N/A')}")
        st.write(f"**Text:** {obj.properties.get('text', 'N/A')}")
        st.write(f"**Description:** {obj.properties.get('description', 'N/A')}")
        st.write(f"**Trace:** {obj.properties.get('trace', 'N/A')}")
        
        # Display image if present
        image_b64 = obj.properties.get('image')
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

client.close()