import base64
import io
import os
from typing import Optional

import streamlit as st
import weaviate
from PIL import Image

from weaviate_v3.block_filters import build_block_filter


DEFAULT_HOST = os.getenv("WEAVIATE_HOST", "172.17.0.2")
DEFAULT_PORT = int(os.getenv("WEAVIATE_PORT", "8080"))
DEFAULT_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
DEFAULT_COLLECTION = os.getenv("WEAVIATE_COLLECTION", "unified_embedding")


@st.cache_resource(show_spinner=False)
def get_client(host: str, port: int, grpc_port: int):
    return weaviate.connect_to_local(host=host, port=port, grpc_port=grpc_port)


def safe_decode_image(image_b64: str) -> Optional[Image.Image]:
    if not image_b64:
        return None
    try:
        image_bytes = base64.b64decode(image_b64)
        return Image.open(io.BytesIO(image_bytes))
    except Exception:
        return None


def render_chunk(idx: int, obj, show_original_text: bool, show_image: bool):
    props = obj.properties or {}
    chunk_type = props.get("block_type", "Unknown")
    page = props.get("page", "N/A")
    filename = props.get("filename", "Unknown")
    chunk_id = props.get("chunk_id") or props.get("id") or ""

    with st.expander(f"{idx}. {filename} | page {page} | {chunk_type}"):
        st.write(f"**UUID:** `{obj.uuid}`")
        if chunk_id:
            st.write(f"**Chunk ID:** `{chunk_id}`")
        st.write(f"**Type:** `{chunk_type}`")
        st.write(f"**Page:** `{page}`")
        st.write(f"**Filename:** `{filename}`")

        text_preview = props.get("text_preview", "")
        text = props.get("text", "")
        trace = props.get("trace", "")

        if text_preview:
            st.write("**Text Preview**")
            st.write(text_preview)

        if show_original_text and text:
            st.write("**Original Text**")
            st.write(text)

        if trace:
            st.write("**Trace**")
            st.code(trace)

        if show_image:
            image_b64 = props.get("images")
            if image_b64:
                image = safe_decode_image(image_b64)
                if image is not None:
                    st.write("**Image**")
                    st.image(image, use_container_width=True)
                else:
                    st.warning("Image exists but could not be decoded")
            else:
                st.caption("No image for this chunk")


def main():
    st.set_page_config(page_title="Unified Embedding Chunk Viewer", layout="wide")
    st.title("Unified Embedding Chunk Viewer")

    with st.sidebar:
        st.header("Connection")
        host = st.text_input("Host", value=DEFAULT_HOST)
        port = st.number_input("HTTP Port", min_value=1, max_value=65535, value=DEFAULT_PORT)
        grpc_port = st.number_input("gRPC Port", min_value=1, max_value=65535, value=DEFAULT_GRPC_PORT)
        collection_name = st.text_input("Collection", value=DEFAULT_COLLECTION)

        st.header("Display")
        page_size = st.slider("Items per page", min_value=5, max_value=200, value=25)
        offset = st.number_input("Offset", min_value=0, value=0, step=page_size)
        show_original_text = st.checkbox("Show original text", value=True)
        show_image = st.checkbox("Show images", value=True)

        st.header("Weaviate filters")
        block_filter = st.selectbox(
            "Block filter preset",
            options=["all", "chunks"],
            index=0,
        )

        st.header("Client-side filters")
        filename_filter = st.text_input("Filename contains", value="").strip().lower()
        text_filter = st.text_input("Text contains", value="").strip().lower()

    try:
        client = get_client(host, int(port), int(grpc_port))
        collection = client.collections.get(collection_name)
    except Exception as exc:
        st.error(f"Failed to connect or get collection: {exc}")
        st.stop()

    try:
        weaviate_filter = build_block_filter("block_type", block_filter)
        total = collection.aggregate.over_all(total_count=True, filters=weaviate_filter)
        st.metric("Total chunks", total.total_count)

        response = collection.query.fetch_objects(
            limit=page_size,
            offset=int(offset),
            filters=weaviate_filter,
            return_properties=[
                "chunk_id",
                "block_type",
                "page",
                "filename",
                "text_preview",
                "text",
                "trace",
                "images",
            ],
        )
    except Exception as exc:
        st.error(f"Query failed: {exc}")
        st.stop()

    filtered = []
    for obj in response.objects:
        props = obj.properties or {}
        filename = str(props.get("filename", "")).lower()
        text = str(props.get("text", "")).lower()
        preview = str(props.get("text_preview", "")).lower()

        if filename_filter and filename_filter not in filename:
            continue
        if text_filter and text_filter not in text and text_filter not in preview:
            continue

        filtered.append(obj)

    st.caption(f"Showing {len(filtered)} items from fetched window ({len(response.objects)} fetched)")

    if not filtered:
        st.info("No chunks match current page/filters")
        return

    for i, obj in enumerate(filtered, start=1):
        render_chunk(i, obj, show_original_text=show_original_text, show_image=show_image)


if __name__ == "__main__":
    main()
