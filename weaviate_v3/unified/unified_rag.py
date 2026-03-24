from typing import Dict, List, Optional

import ollama
import requests
import weaviate
from weaviate.classes.query import MetadataQuery

from weaviate_v3.block_filters import build_block_filter


class UnifiedChatRAG:
    def __init__(
        self,
        collection_name: str,
        weaviate_host: str,
        ollama_model: str,
        multimodal: bool = False,
        weaviate_port: int = 8080,
        weaviate_grpc_port: int = 50051,
        summarize_threshold_tokens: int = 70,
        ollama_host: str = "172.17.0.6",
        ollama_port: int = 11434,
        multi2vec_host: str = "172.17.0.7",
        multi2vec_port: int = 8080,
        default_block_filter: Optional[str] = None,
    ):
        self.client = weaviate.connect_to_local(
            host=weaviate_host,
            port=weaviate_port,
            grpc_port=weaviate_grpc_port,
        )
        self.collection_name = collection_name
        self.ollama_model = ollama_model
        self.multimodal = multimodal
        self.summarize_threshold_tokens = summarize_threshold_tokens
        self.ollama_host = ollama_host
        self.ollama_port = ollama_port
        self.ollama_base_url = f"http://{self.ollama_host}:{self.ollama_port}"
        self.ollama_client = ollama.Client(host=self.ollama_base_url)
        self.multi2vec_host = multi2vec_host
        self.multi2vec_port = multi2vec_port
        self.multi2vec_base_url = f"http://{self.multi2vec_host}:{self.multi2vec_port}"
        self.messages: List[Dict[str, str]] = []
        self.default_block_filter = default_block_filter

        available_collections = self.client.collections.list_all()
        if self.collection_name not in available_collections:
            available = ", ".join(sorted(available_collections.keys()))
            raise ValueError(
                f"Collection '{self.collection_name}' not found. Available: {available}"
            )

        self.chunks = self.client.collections.get(self.collection_name)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return len((text or "").split())

    def _embed_query(self, query: str) -> List[float]:
        response = requests.post(
            f"{self.multi2vec_base_url}/vectorize",
            json={"texts": [query], "images": []},
            headers={"Content-Type": "application/json"},
            timeout=60,
        )
        response.raise_for_status()

        data = response.json()
        text_vectors = data.get("textVectors") or []
        if not text_vectors or not isinstance(text_vectors[0], list):
            raise RuntimeError("multi2vec-clip returned no textVectors for the query")
        return text_vectors[0]

    def _summarize_if_needed(self, user_message: str) -> Dict[str, Optional[str]]:
        token_count = self._estimate_tokens(user_message)
        info = {
            "auto_summarized": False,
            "original_tokens": token_count,
            "original_prompt": user_message,
            "effective_prompt": user_message,
        }

        if token_count <= self.summarize_threshold_tokens:
            return info

        summarize_prompt = (
            f"Rewrite the user query in <= {self.summarize_threshold_tokens} tokens "
            "while preserving all key intent, constraints, and technical terms. "
            "Return only the rewritten query.\n\n"
            f"User query:\n{user_message}\n"
        )
        try:
            response = self.ollama_client.generate(model=self.ollama_model, prompt=summarize_prompt)
            summarized = (response.get("response") or "").strip()
        except Exception:
            summarized = ""

        if not summarized:
            summarized = " ".join(user_message.split()[: self.summarize_threshold_tokens]).strip()

        info["auto_summarized"] = True
        info["effective_prompt"] = summarized
        return info

    @staticmethod
    def _apply_modality_rerank(sources: List[Dict], modality_balance: float) -> List[Dict]:
        """Query-time rerank only: 0.0 favors text-heavy chunks, 1.0 favors image chunks."""
        if not sources:
            return sources

        modality_balance = max(0.0, min(1.0, modality_balance))
        if abs(modality_balance - 0.5) < 1e-6:
            return sources

        total = len(sources)
        rescored = []
        for idx, src in enumerate(sources):
            distance = src.get("weaviate_distance")
            if isinstance(distance, (int, float)):
                base_score = -float(distance)
            else:
                base_score = float(total - idx)

            has_image = 1.0 if src.get("images") else 0.0
            has_text = 1.0 if (src.get("text_preview") or src.get("text")) else 0.0

            image_boost = modality_balance * has_image * 0.4
            text_boost = (1.0 - modality_balance) * has_text * 0.4
            score = base_score + image_boost + text_boost
            rescored.append((score, src))

        rescored.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in rescored]

    def retrieve_context(
        self,
        query: str,
        top_k: int = 5,
        query_alpha: float = 0.7,
        modality_balance: float = 0.5,
        block_filter: Optional[str] = None,
    ) -> Dict[str, Optional[List[Dict]]]:
        return_properties = [
            "chunk_id",
            "block_type",
            "page",
            "text_preview",
            "text",
            "trace",
            "filename",
            "images",
        ]

        block_filter = block_filter if block_filter is not None else self.default_block_filter
        filters = build_block_filter("block_type", block_filter)
        retrieval_mode = "near_vector_runtime_embedding"
        retrieval_warning = None

        try:
            query_vector = self._embed_query(query)
            response = self.chunks.query.near_vector(
                near_vector=query_vector,
                limit=top_k,
                filters=filters,
                return_properties=return_properties,
                return_metadata=MetadataQuery(distance=True),
            )
        except Exception as exc:
            retrieval_mode = "bm25_fallback"
            retrieval_warning = (
                "Runtime multi2vec vector search failed, so lexical BM25 fallback was used. "
                f"Original error: {exc}"
            )
            response = self.chunks.query.bm25(
                query=query,
                limit=top_k,
                filters=filters,
                return_properties=return_properties,
                return_metadata=MetadataQuery(distance=True),
            )

        results = []
        for obj in response.objects:
            results.append(
                {
                    **(obj.properties or {}),
                    "weaviate_distance": obj.metadata.distance if obj.metadata else None,
                }
            )
        results = self._apply_modality_rerank(results, modality_balance)
        return {
            "sources": results,
            "retrieval_mode": retrieval_mode,
            "retrieval_warning": retrieval_warning,
        }

    @staticmethod
    def format_context(retrieved_docs: List[Dict], include_image_info: bool = False) -> str:
        if not retrieved_docs:
            return ""

        context_parts = ["Here is relevant context from the knowledge base:\n"]
        for i, doc in enumerate(retrieved_docs, start=1):
            context_parts.append(f"\n[Document {i}]")
            context_parts.append(f"Type: {doc.get('block_type', 'Unknown')}")
            context_parts.append(f"Page: {doc.get('page', 'N/A')}")
            context_parts.append(f"Filename: {doc.get('filename', 'Unknown')}")
            if doc.get("trace"):
                context_parts.append(f"Trace: {doc.get('trace')}")

            if include_image_info and doc.get("images"):
                context_parts.append(f"[Image {i} is attached - refer to it when answering]")

            text_preview = doc.get("text_preview") or ""
            text = doc.get("text") or ""
            content = text_preview if text_preview else text
            context_parts.append(f"Content: {content}")
            context_parts.append("")

        return "\n".join(context_parts)

    def chat(
        self,
        user_message: str,
        use_rag: bool = True,
        top_k: int = 3,
        query_alpha: float = 0.7,
        modality_balance: float = 0.5,
        system_prompt: Optional[str] = None,
        return_sources: bool = True,
        block_filter: Optional[str] = None,
    ) -> Dict:
        block_filter = block_filter if block_filter is not None else self.default_block_filter
        summary_info = self._summarize_if_needed(user_message)
        effective_message = summary_info["effective_prompt"] or user_message

        history_context = ""
        if self.messages:
            recent_history = self.messages[-2:]
            history_context = "Previous conversation context: " + "; ".join(
                [f"Q: {h['question']} A: {h['answer'][:120]}" for h in recent_history]
            )

        sources: List[Dict] = []
        context = ""
        images = []
        retrieval_mode = None
        retrieval_warning = None

        if use_rag:
            retrieval = self.retrieve_context(
                effective_message,
                top_k=top_k,
                query_alpha=query_alpha,
                modality_balance=modality_balance,
                block_filter=block_filter,
            )
            sources = retrieval.get("sources") or []
            retrieval_mode = retrieval.get("retrieval_mode")
            retrieval_warning = retrieval.get("retrieval_warning")
            context = self.format_context(sources, include_image_info=self.multimodal)
            if self.multimodal:
                images = [doc["images"] for doc in sources if doc.get("images")]

        if system_prompt is None:
            system_prompt = (
                "You are a helpful Open RAN assistant."
            )

        full_prompt = (
            f"User Question: {effective_message}\n\n"
            f"{context}\n\n"
            f"{system_prompt}\n\n"
#            f"{history_context}\n\n"
            "Answer:"
        )

        if self.multimodal and images:
            response = self.ollama_client.chat(
                model=self.ollama_model,
                messages=[
                    {
                        "role": "user",
                        "content": full_prompt,
                        "images": images,
                    }
                ],
            )
            answer = response["message"]["content"]
        else:
            response = self.ollama_client.generate(model=self.ollama_model, prompt=full_prompt)
            answer = response["response"]

        self.messages.append(
            {
                "question": user_message,
                "effective_question": effective_message,
                "answer": answer,
            }
        )

        result = {
            "answer": answer,
            "summary_info": summary_info,
            "retrieval_mode": retrieval_mode,
            "retrieval_warning": retrieval_warning,
        }
        if return_sources:
            result["sources"] = sources
        return result

    def clear_history(self):
        self.messages = []

    def get_history(self) -> List[Dict[str, str]]:
        return self.messages.copy()

    def close(self):
        self.client.close()
