import atexit
import base64
import io

import streamlit as st
from PIL import Image

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from  weaviate_v3.unified.unified_rag import UnifiedChatRAG


st.set_page_config(
    page_title="Unified Multimodal RAG Assistant",
    page_icon="📡",
    layout="wide",
)


def _show_sources(sources):
    with st.expander("View Sources"):
        for i, source in enumerate(sources, 1):
            st.markdown(f"**Source {i}**")
            col1, col2, col3, col4 = st.columns(4)
            col1.text(f"Type: {source.get('block_type', 'Unknown')}")
            col2.text(f"Page: {source.get('page', 'Unknown')}")
            col3.text(f"File: {source.get('filename', 'Unknown')}")
            distance = source.get("weaviate_distance")
            col4.text(
                f"Distance: {distance:.4f}" if isinstance(distance, float) else "Distance: N/A"
            )

            preview = source.get("text_preview") or source.get("text") or ""
            st.text(preview[:400] + "..." if len(preview) > 400 else preview)

            image_b64 = source.get("images")
            if image_b64:
                try:
                    image_bytes = base64.b64decode(image_b64)
                    image = Image.open(io.BytesIO(image_bytes))
                    st.image(image, use_container_width=True)
                except Exception:
                    st.caption("Image exists but could not be decoded.")

            if source.get("trace"):
                st.caption(f"Trace: {source['trace']}")
            st.divider()


if "chat_rag" not in st.session_state:
    st.session_state.chat_rag = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "initialized" not in st.session_state:
    st.session_state.initialized = False

with st.sidebar:
    st.title("Settings")
    weaviate_host = st.text_input("Weaviate Host", "172.17.0.5")
    weaviate_port = st.number_input("Weaviate HTTP Port", min_value=1, max_value=65535, value=8080)
    weaviate_grpc_port = st.number_input(
        "Weaviate gRPC Port", min_value=1, max_value=65535, value=50051
    )
    collection_name = st.text_input("Collection Name", "Unified_embedding")
    ollama_model = st.text_input("Ollama Model", "llama3.2")
    ollama_host = st.text_input("Ollama Host", "172.17.0.6")
    ollama_port = st.number_input("Ollama Port", min_value=1, max_value=65535, value=11434)
    multi2vec_host = st.text_input("multi2vec Host", "172.17.0.7")
    multi2vec_port = st.number_input("multi2vec Port", min_value=1, max_value=65535, value=8080)
    multimodal = st.checkbox("Multimodal Chat (attach retrieved images)", value=True)

    st.divider()
    use_rag = st.checkbox("Enable RAG", value=True)
    num_results = st.slider("Chunks to retrieve", 1, 10, 3)
    query_alpha = st.slider(
        "Lexical fallback balance",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="Kept for compatibility; primary retrieval now uses runtime multi2vec embeddings, with BM25 fallback if vector retrieval fails.",
    )
    modality_balance_pct = st.slider(
        "Unified modality preference (text <-> image)",
        min_value=0,
        max_value=100,
        value=50,
        step=5,
        help="Query-time rerank preference only. 0=text-heavy, 100=image-heavy.",
    )
    summarize_threshold = st.number_input(
        "Auto-summarize if tokens >", min_value=1, max_value=500, value=70
    )
    block_filter_label = st.selectbox("Block Filter", ["all", "chunks"], index=0)
    block_filter = None if block_filter_label == "all" else block_filter_label

    if st.button("Initialize System", type="primary"):
        with st.spinner("Initializing unified RAG system..."):
            try:
                st.session_state.chat_rag = UnifiedChatRAG(
                    collection_name=collection_name,
                    weaviate_host=weaviate_host,
                    ollama_model=ollama_model,
                    multimodal=multimodal,
                    weaviate_port=int(weaviate_port),
                    weaviate_grpc_port=int(weaviate_grpc_port),
                    summarize_threshold_tokens=int(summarize_threshold),
                    ollama_host=ollama_host,
                    ollama_port=int(ollama_port),
                    multi2vec_host=multi2vec_host,
                    multi2vec_port=int(multi2vec_port),
                    default_block_filter=block_filter,
                )
                st.session_state.initialized = True
                st.success("System initialized")
            except Exception as exc:
                st.error(f"Initialization failed: {exc}")

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        if st.session_state.chat_rag:
            st.session_state.chat_rag.clear_history()
        st.rerun()

st.title("Unified Multimodal RAG Assistant")
st.caption(
    "This interface queries the multimodal `unified_embedding` collection and auto-summarizes long prompts."
)

if not st.session_state.initialized:
    st.info("Configure settings in the sidebar and click 'Initialize System'.")
    st.stop()

if st.session_state.chat_rag:
    atexit.register(lambda: st.session_state.chat_rag.close())

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        summary_info = message.get("summary_info")
        if summary_info and summary_info.get("auto_summarized"):
            st.info(
                f"Input exceeded {st.session_state.chat_rag.summarize_threshold_tokens} tokens, so it was auto-summarized."
            )
            st.code(summary_info.get("effective_prompt", ""), language="text")

        if message["role"] == "assistant" and "sources" in message:
            _show_sources(message["sources"])

if prompt := st.chat_input("Ask about O-RAN specs..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer..."):
            try:
                result = st.session_state.chat_rag.chat(
                    user_message=prompt,
                    use_rag=use_rag,
                    top_k=num_results,
                    query_alpha=float(query_alpha),
                    modality_balance=float(modality_balance_pct) / 100.0,
                    return_sources=True,
                    block_filter=block_filter,
                )

                answer = result.get("answer", "")
                sources = result.get("sources", [])
                summary_info = result.get("summary_info", {})

                st.markdown(answer)

                if summary_info.get("auto_summarized"):
                    threshold = st.session_state.chat_rag.summarize_threshold_tokens
                    st.info(
                        f"Input exceeded {threshold} tokens, so it was auto-summarized before retrieval/chat."
                    )
                    st.markdown("**Summarized prompt used for chat:**")
                    st.code(summary_info.get("effective_prompt", ""), language="text")

                if sources:
                    _show_sources(sources)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "summary_info": summary_info,
                    }
                )
            except Exception as exc:
                error_text = f"Error: {exc}"
                st.error(error_text)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_text,
                    }
                )
