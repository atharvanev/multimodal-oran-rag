import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import MetadataQuery
import ollama
from typing import List, Dict, Optional
import json
import requests

from phoenix.otel import register
from opentelemetry.instrumentation.ollama import OllamaInstrumentor
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.trace import Status, StatusCode
import phoenix as px

from weaviate_v3.block_filters import build_block_filter


class ChatRAG:
    def __init__(
            self,
            collection_name: str,
            weaviate_host: str,
            weaviate_port: int,
            ollama_model: str,
            ollama_port: int,
            enable_phoenix: bool = True,
            phoenix_endpoint: str = "http://localhost:6006/v1/traces",
            weaviate_grpc_port: int = 50051,
            ollama_host: str = "172.17.0.6",
            default_block_filter: Optional[str] = None):

        self.enable_phoenix = enable_phoenix

        if self.enable_phoenix:
            self._setup_phoenix_(phoenix_endpoint)

        self.tracer = trace.get_tracer(__name__)

        with self.tracer.start_as_current_span("chat_rag_initialization") as span:
            span.set_attribute("weaviate_host", weaviate_host)
            span.set_attribute("weaviate_port", weaviate_port)
            span.set_attribute("collection_name", collection_name)
            span.set_attribute("ollama_model", ollama_model)
            span.set_attribute("ollama_port", ollama_port)

            try:
                self.weaviate_host = weaviate_host
                self.weaviate_port = weaviate_port
                self.weaviate_grpc_port = weaviate_grpc_port
                self.ollama_host = ollama_host
                self.ollama_port = ollama_port
                self.ollama_base_url = f"http://{self.ollama_host}:{self.ollama_port}"
                self.ollama_client = ollama.Client(host=self.ollama_base_url)
                self.embedding_model = "nomic-embed-text"
                self.client = weaviate.connect_to_local(
                    host=weaviate_host,
                    port=self.weaviate_port,
                    grpc_port=self.weaviate_grpc_port,
                )

                self.collection_name = collection_name
                self.ollama_model = ollama_model
                self.messages: List[Dict[str, str]] = []
                self.default_block_filter = default_block_filter

                if self.collection_name not in self.client.collections.list_all():
                    span.add_event("Creating new Weaviate collection via REST API")

                    schema = {
                        "class": self.collection_name,
                        "vectorizer": "text2vec-ollama",
                        "moduleConfig": {
                            "text2vec-ollama": {
                                "apiEndpoint": self.ollama_base_url,
                                "model": "nomic-embed-text"
                            },
                            "generative-ollama": {
                                "apiEndpoint": self.ollama_base_url,
                                "model": self.ollama_model
                            }
                        },
                        "properties": [
                            {"name": "type", "dataType": ["text"], "moduleConfig": {"text2vec-ollama": {"skip": True}}},
                            {"name": "page", "dataType": ["int"], "moduleConfig": {"text2vec-ollama": {"skip": True}}},
                            {"name": "description", "dataType": ["text"], "moduleConfig": {"text2vec-ollama": {"skip": False}}},
                            {"name": "text", "dataType": ["text"], "moduleConfig": {"text2vec-ollama": {"skip": False}}},
                            {"name": "trace", "dataType": ["text"], "moduleConfig": {"text2vec-ollama": {"skip": False}}},
                            {"name": "filename", "dataType": ["text"], "moduleConfig": {"text2vec-ollama": {"skip": False}}},
                            {"name": "images", "dataType": ["blob"], "moduleConfig": {"text2vec-ollama": {"skip": True}}}
                        ]
                    }

                    r = requests.post(
                        f"http://{self.weaviate_host}:{self.weaviate_port}/v1/schema",
                        json=schema,
                    )

                    if r.status_code != 200:
                        raise Exception(f"Failed to create collection: {r.status_code} - {r.text}")

                    print(f"✓ Collection '{self.collection_name}' created successfully")
                else:
                    span.add_event("Using existing Weaviate collection")

                self.chunks = self.client.collections.get(self.collection_name)
                span.add_event("ChatRAG initialization complete")

            except Exception as e:
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                raise

    def _setup_phoenix_(self, endpoint: str):
        try:
            px.launch_app()

            tracer_provider = TracerProvider()
            tracer_provider.add_span_processor(
                SimpleSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
            )

            trace.set_tracer_provider(tracer_provider)
            register()
            OllamaInstrumentor().instrument()

            print(f"✓ Phoenix tracing enabled at {endpoint}")
            print("✓ View traces at http://localhost:6006")
        except Exception as e:
            print(f"Warning: Could not setup Phoenix tracing: {e}")
            self.enable_phoenix = False

    def _embed_query(self, query: str) -> List[float]:
        with self.tracer.start_as_current_span("embed_query") as span:
            span.set_attribute("embedding_model", self.embedding_model)
            span.set_attribute("ollama_base_url", self.ollama_base_url)

            embed_url = f"{self.ollama_base_url}/api/embed"
            legacy_embeddings_url = f"{self.ollama_base_url}/api/embeddings"
            errors = []

            try:
                response = requests.post(
                    embed_url,
                    json={"model": self.embedding_model, "input": query},
                    timeout=60,
                )
                response.raise_for_status()
                data = response.json()
                embeddings = data.get("embeddings") or []
                if embeddings and isinstance(embeddings[0], list):
                    span.set_attribute("embedding_dimensions", len(embeddings[0]))
                    span.set_attribute("embedding_api", "/api/embed")
                    return embeddings[0]
                errors.append(f"{embed_url} returned no embeddings")
            except Exception as exc:
                errors.append(f"{embed_url}: {exc}")

            try:
                response = requests.post(
                    legacy_embeddings_url,
                    json={"model": self.embedding_model, "prompt": query},
                    timeout=60,
                )
                response.raise_for_status()
                data = response.json()
                embedding = data.get("embedding")
                if embedding:
                    span.set_attribute("embedding_dimensions", len(embedding))
                    span.set_attribute("embedding_api", "/api/embeddings")
                    return embedding
                errors.append(f"{legacy_embeddings_url} returned no embedding")
            except Exception as exc:
                errors.append(f"{legacy_embeddings_url}: {exc}")

            error_message = "Failed to embed query via current Ollama host. " + " | ".join(errors)
            span.set_status(trace.Status(trace.StatusCode.ERROR, error_message))
            raise RuntimeError(error_message)

    def add_documents(self, documents: List[Dict], batch_size: int = 200):
        with self.tracer.start_as_current_span("add_documents") as span:
            span.set_attribute("document_count", len(documents))
            span.set_attribute("batch_size", batch_size)

            with self.chunks.batch.fixed_size(batch_size) as batch:
                for idx, doc in enumerate(documents):
                    properties = {
                        "type": doc.get("type") or doc.get("type", "Unknown"),
                        "page": doc.get("page", "Unknown"),
                        "description": doc.get("description") or doc.get("Description", ""),
                        "text": doc.get("text") or doc.get("Text", ""),
                        "trace": doc.get("trace") or doc.get("Trace", ""),
                        "filename": doc.get("filename", ""),
                    }

                    if doc.get("images") or doc.get("image"):
                        properties["images"] = doc.get("images") or doc.get("image")

                    batch.add_object(properties)

                    if batch.number_errors > 10:
                        print("Batch import stopped due to excessive errors.")
                        break

            failed_objects = self.chunks.batch.failed_objects
            if failed_objects:
                span.set_attribute("failed_imports", len(failed_objects))
                span.add_event("import_failures", {"count": len(failed_objects)})
                print(f"Number of failed imports: {len(failed_objects)}")
                print(f"First failed object: {failed_objects[0]}")
            else:
                span.add_event("All documents imported successfully")

    def retrieve_context(
        self,
        query: str,
        top_k: int = 5,
        query_alpha: float = 0.7,
        block_filter: Optional[str] = None,
    ) -> List[Dict]:
        with self.tracer.start_as_current_span("retrieve_context") as span:
            block_filter = block_filter if block_filter is not None else self.default_block_filter
            span.set_attribute("query", query)
            span.set_attribute("top_k", top_k)
            span.set_attribute("query_alpha", query_alpha)
            if block_filter:
                span.set_attribute("block_filter", block_filter)

            return_properties = ["text", "type", "page", "description", "trace", "image", "filename"]
            filters = build_block_filter("type", block_filter)

            retrieval_method = "hybrid_runtime_embedding"
            try:
                query_vector = self._embed_query(query)
                response = self.chunks.query.hybrid(
                    query=query,
                    vector=query_vector,
                    limit=top_k,
                    alpha=query_alpha,
                    filters=filters,
                    return_properties=return_properties,
                    return_metadata=MetadataQuery(distance=True),
                )
            except Exception as exc:
                retrieval_method = "bm25_fallback"
                span.add_event("vector_retrieval_failed", {"error": str(exc)})
                response = self.chunks.query.bm25(
                    query=query,
                    limit=top_k,
                    filters=filters,
                    return_properties=return_properties,
                    return_metadata=MetadataQuery(distance=True),
                )

            span.set_attribute("retrieval_method", retrieval_method)

            results = []
            for obj in response.objects:
                doc = {
                    **obj.properties,
                    "weaviate_distance": obj.metadata.distance if obj.metadata else None
                }
                if not doc.get("block_type") and doc.get("type"):
                    doc["block_type"] = doc["type"]
                results.append(doc)

            span.set_attribute("results_count", len(results))
        return results

    def format_context(self, retrieved_docs: List[Dict]) -> str:
        if not retrieved_docs:
            return ""

        with self.tracer.start_as_current_span("format_context") as span:
            span.set_attribute("document_count", len(retrieved_docs))

            context_parts = ["Here is relevant context from the knowledge base:\n"]
            for i, doc in enumerate(retrieved_docs, 1):
                context_parts.append(f"\n[Document {i}]")

                if doc.get('type'):
                    context_parts.append(f"Type: {doc['type']}")
                if doc.get('page'):
                    context_parts.append(f"Page: {doc['page']}")
                if doc.get('trace'):
                    context_parts.append(f"Trace: {doc['trace']}")
                if doc.get('filename'):
                    context_parts.append(f"Filename: {doc['filename']}")

                context_parts.append(f"Content: {doc.get('text', '')}")

                if doc.get('description'):
                    context_parts.append(f"Description: {doc['description']}")

                context_parts.append("")
            return "\n".join(context_parts)

    def chat(
            self,
            user_message: str,
            use_rag: bool = True,
            top_k: int = 3,
            query_alpha: float = 0.7,
            system_prompt: Optional[str] = None,
            return_sources: bool = True,
            block_filter: Optional[str] = None) -> Dict:

        with self.tracer.start_as_current_span("chat") as span:
            block_filter = block_filter if block_filter is not None else self.default_block_filter
            span.set_attribute("user_message", user_message)
            span.set_attribute("use_rag", use_rag)
            span.set_attribute("top_k", top_k)
            span.set_attribute("query_alpha", query_alpha)
            span.set_attribute("ollama_model", self.ollama_model)
            if block_filter:
                span.set_attribute("block_filter", block_filter)

            history_context = ""
            if self.messages:
                recent_history = self.messages[-2:]
                history_context = "Previous conversation context: " + "; ".join([
                    f"Q: {h['question']} A: {h['answer'][:100]}"
                    for h in recent_history
                ])

            sources = []
            context = ""

            if use_rag:
                sources = self.retrieve_context(
                    user_message,
                    top_k=top_k,
                    query_alpha=query_alpha,
                    block_filter=block_filter,
                )
                context = self.format_context(sources)
                span.set_attribute("context_length", len(context))

            if system_prompt is None:
                system_prompt = "You are a helpful Open Radio Access Network assistant and expert helping users with their questions grounded in provided context."

            full_prompt = f"{system_prompt}\n\n{history_context}\n\n{context}\n\nUser Question: {user_message}\n\nAnswer:"
            span.set_attribute("prompt_length", len(full_prompt))

            response = self.ollama_client.generate(
                model=self.ollama_model,
                prompt=full_prompt
            )
            answer = response['response']

            span.set_attribute("response_length", len(answer))

            self.messages.append({
                'question': user_message,
                'answer': answer
            })

            span.add_event("conversation_updated", {"total_turns": len(self.messages)})

            result = {'answer': answer}
            if return_sources:
                result['sources'] = sources
            return result

    def clear_history(self):
        with self.tracer.start_as_current_span("clear_history") as span:
            previous_length = len(self.messages)
            self.messages = []
            span.set_attribute("cleared_messages", previous_length)
            print("Conversation history cleared")

    def get_history(self) -> List[Dict[str, str]]:
        return self.messages.copy()

    def close(self):
        with self.tracer.start_as_current_span("close_connection") as span:
            self.client.close()
            span.add_event("weaviate_connection_closed")


if __name__ == "__main__":
    chat_rag = ChatRAG(
        weaviate_host="172.17.0.6",
        weaviate_port=8080,
        collection_name="Documents",
        ollama_model="llama3.2",
        ollama_port=11434
    )
