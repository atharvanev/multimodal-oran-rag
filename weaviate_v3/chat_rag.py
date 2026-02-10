import weaviate
from weaviate.classes.config  import Configure,Property, DataType
from weaviate.classes.query import MetadataQuery
import ollama
from typing import List, Dict, Optional
import json

from phoenix.otel import register
from opentelemetry.instrumentation.ollama import OllamaInstrumentor
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.trace import Status, StatusCode
import phoenix as px



class ChatRAG:
    def __init__(
            self, collection_name: str, 
            weaviate_host: str, 
            ollama_model: str, 
            multimodal: bool = False,
            enable_phoenix: bool = True,
            phoenix_endpoint: str = "http://localhost:6006/v1/traces"):
        
        self.enable_phoenix = enable_phoenix
        
        if self.enable_phoenix:
            self._setup_phoenix_(phoenix_endpoint)
        
        self.tracer = trace.get_tracer(__name__)

        with self.tracer.start_as_current_span("chat_rag_initialization") as span:
            span.set_attribute("weaviate_host", weaviate_host)
            span.set_attribute("collection_name", collection_name)
            span.set_attribute("ollama_model", ollama_model)
            span.set_attribute("multimodal", multimodal)

            try:
                self.weaviate_host = weaviate_host
                self.client = weaviate.connect_to_local(
                    host=weaviate_host,  
                    port=8080,
                    grpc_port=50051,
                )
                
                self.collection_name = collection_name
                self.ollama_model = ollama_model
                self.messages: List[Dict[str, str]] = []
                self.multimodal = multimodal

                if self.collection_name not in self.client.collections.list_all():
                    span.add_event("Creating new Weaviate collection via REST API")
                    
                    # Create collection via REST API with proper skip settings
                    schema = {
                        "class": self.collection_name,
                        "vectorizer": "text2vec-ollama",
                        "moduleConfig": {
                            "text2vec-ollama": {
                                "apiEndpoint": "http://172.17.0.4:11434",
                                "model": "nomic-embed-text"
                            },
                            "generative-ollama": {
                                "apiEndpoint": "http://172.17.0.4:11434",
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
                            {"name": "image", "dataType": ["blob"], "moduleConfig": {"text2vec-ollama": {"skip": True}}}
                        ]
                    }
                    
                    import requests
                    r = requests.post(f"http://{self.weaviate_host}:8080/v1/schema", json=schema)
                    
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
        """Setup Phoenix observability."""
        try:
            # Launch Phoenix in the background
            px.launch_app()
            
            # Setup tracer provider and set it globally
            tracer_provider = TracerProvider()
            tracer_provider.add_span_processor(
                SimpleSpanProcessor(OTLPSpanExporter(endpoint=endpoint))
            )
            
            # Set as global tracer provider (OpenTelemetry standard way)
            trace.set_tracer_provider(tracer_provider)
            
            # Register with Phoenix (no arguments needed)
            register()
            
            # Auto-instrument Ollama
            OllamaInstrumentor().instrument()
            
            print(f"✓ Phoenix tracing enabled at {endpoint}")
            print("✓ View traces at http://localhost:6006")
        except Exception as e:
            print(f"Warning: Could not setup Phoenix tracing: {e}")
            self.enable_phoenix = False

    def add_documents(
            self, documents: List[Dict], 
            batch_size: int = 200):
        
        with self.tracer.start_as_current_span("add_documents") as span:
            span.set_attribute("document_count", len(documents))
            span.set_attribute("batch_size", batch_size)
        
            
            with self.chunks.batch.fixed_size(batch_size) as batch:
                for idx,doc in enumerate(documents):
                    properties = {
                        "type": doc.get("type") or doc.get("block_type", "Unknown"),
                        "page": doc.get("page", "Unknown"),
                        "description": doc.get("description") or doc.get("Description", ""),
                        "text": doc.get("text") or doc.get("Text", ""),
                        "trace": doc.get("trace") or doc.get("Trace", ""),
                    }

                    # Handle image properly
                    if doc["images"]:
                        properties["image"] = doc["images"]
                
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
            query:str, 
            top_k:int=5
            ) -> List[Dict]:
        

        with self.tracer.start_as_current_span("retrieve_context") as span:
            span.set_attribute("query", query)
            span.set_attribute("top_k", top_k)
            span.set_attribute("retrieval_method", "hybrid")
            span.set_attribute("alpha", 0.7)

            return_properties = ["text", "page", "type", "description", "trace", "image","filename"]

            response = self.chunks.query.hybrid(
                query=query,
                limit=top_k,
                return_properties=return_properties,
                return_metadata=MetadataQuery(distance=True),
                alpha=0.7
            )

            results = []

            for obj in response.objects:
                doc = {
                    **obj.properties,
                    "weaviate_distance": obj.metadata.distance if obj.metadata else None
                }
                results.append(doc)

            chunks_summary = [
                    {
                       "page": r.get("page"),
                       "type": r.get("type"),
                       "snippet": (r.get("text","")[:300] + "...") if len(r.get("text","") or "") > 300 else r.get("text",""),
                       "distance": r.get("weaviate_distance")
                   }
                for r in results
               ]
            span.set_attribute("results_count", len(results))
        return results
    
    def format_context(self, retrieved_docs: List[Dict], include_image_info: bool = False) -> str:
            """Format retrieved documents into context string with all metadata."""
            if not retrieved_docs:
                return ""
            
            with self.tracer.start_as_current_span("format_context") as span:
                span.set_attribute("document_count", len(retrieved_docs))
                span.add_event("Formatting retrieved documents into context string")

                context_parts = ["Here is relevant context from the knowledge base:\n"]
                for i, doc in enumerate(retrieved_docs, 1):
                    context_parts.append(f"\n[Document {i}]")
                    
                    # Add metadata
                    if doc.get('type'):
                        context_parts.append(f"Type: {doc['type']}")
                    if doc.get('page'):
                        context_parts.append(f"Page: {doc['page']}")
                    if doc.get('trace'):
                        context_parts.append(f"Trace: {doc['trace']}")
                    if doc.get('filename'):
                        context_parts.append(f"Filename: {doc['filename']}")
                    
                    # Note if image is available (for multimodal context)
                    if include_image_info and doc.get('image'):
                        context_parts.append(f"[Image {i} is attached - refer to it when answering]")
                    
                    # Add main text
                    context_parts.append(f"Content: {doc.get('text', '')}")
                    
                    # Add description if available
                    if doc.get('description'):
                        context_parts.append(f"Description: {doc['description']}")
                    
                    context_parts.append("")  # Empty line between documents
                return "\n".join(context_parts)

    def chat(
        self,
        user_message: str,
        use_rag: bool = True,
        top_k: int = 3,
        system_prompt: Optional[str] = None,
        return_sources: bool = True) -> Dict:
        

        with self.tracer.start_as_current_span("chat") as span:
            span.set_attribute("user_message", user_message)
            span.set_attribute("use_rag", use_rag)
            span.set_attribute("top_k", top_k)
            span.set_attribute("ollama_model", self.ollama_model)
            span.set_attribute("multimodal", self.multimodal)

            # Build conversation history context
            history_context = ""
            if self.messages:
                recent_history = self.messages[-2:]  # Last 2 turns
                history_context = "Previous conversation context: " + "; ".join([
                    f"Q: {h['question']} A: {h['answer'][:100]}"
                    for h in recent_history
                ])
            
            sources = []
            context = ""
            images = []

            if use_rag:
                sources = self.retrieve_context(user_message, top_k=top_k)
                context = self.format_context(sources, include_image_info=self.multimodal)
                span.set_attribute("context_length", len(context))
            
                if self.multimodal:
                    for doc in sources:
                        if doc.get("image"):
                            images.append(doc["image"])
                    span.set_attribute("image_count", len(images))
                    span.add_event("multimodal_images_extracted", {"count": len(images)})

            if system_prompt is None:
                system_prompt = "You are a helpful Open Radio Acess Network asisstant and expert helping users with their questions grounded in provided context."
            
            full_prompt = f"{system_prompt}\n\n{history_context}\n\n{context}\n\nUser Question: {user_message}\n\nAnswer:"
            span.set_attribute("prompt_length", len(full_prompt))

            if self.multimodal and images:
                # Use chat API for multimodal
                span.add_event("using_multimodal_chat")
                response = ollama.chat(
                    model=self.ollama_model,
                    messages=[{
                        'role': 'user',
                        'content': full_prompt,
                        'images': images  # Pass base64 images directly
                    }]
                )
                answer = response['message']['content']
            else:
                # Use generate API for text-only
                span.add_event("using_text_only_generation")
                response = ollama.generate(
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

            result = {
                'answer': answer
            }
            if return_sources:
                result['sources'] = sources
            return result
    
    def clear_history(self):
        """Clear the conversation history."""
        with self.tracer.start_as_current_span("clear_history") as span:
            previous_length = len(self.messages)
            self.messages = []
            span.set_attribute("cleared_messages", previous_length)
            print("Conversation history cleared")
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get the current conversation history."""
        return self.messages.copy()
    
    def close(self):
        """Close Weaviate connection."""
        with self.tracer.start_as_current_span("close_connection") as span:
            self.client.close()
            span.add_event("weaviate_connection_closed")


if __name__ == "__main__":
    # Initialize the chat system
    chat_rag = ChatRAG(
        weaviate_host="172.17.0.4",
        collection_name="Documents",
        ollama_model="llama3.2"
    )
    
    # Example: Add some documents to Weaviate
    sample_docs = [
        {
            "text": "Python is a high-level programming language known for its simplicity and readability.",
            "page": 1,
            "type": "text",
            "description": "Introduction to Python",
            "trace": "doc_001",
            "images": None
        },
        {
            "text": "Machine learning is a subset of AI that enables systems to learn from data.",
            "page": 5,
            "type": "text",
            "description": "ML Basics",
            "trace": "doc_002",
            "images": None
        },
        {
            "text": "Weaviate is a vector database that enables semantic search capabilities.",
            "page": 12,
            "type": "text",
            "description": "Vector Databases",
            "trace": "doc_003",
            "images": None
        }
    ]
    
    # Uncomment to add documents (only needed once)
    #chat_rag.add_documents(sample_docs)
    
    # Chat with RAG
    print("=== Chat with RAG ===")
    result = chat_rag.chat(
        "What is Python?",
        use_rag=True,
        top_k=2
    )
    print(f"Answer: {result['answer']}\n")
    
    print("Sources:")
    for i, source in enumerate(result['sources'], 1):
        print(f"{i}. Page {source.get('page')}: {source.get('text')[:100]}...")
    
    print("\n" + "="*60 + "\n")
    
    # Follow-up question (history is maintained)
    result2 = chat_rag.chat(
        "Can you tell me more about its uses?",
        use_rag=True
    )
    print(f"Answer: {result2['answer']}\n")
    
    # Chat without RAG (just using history)
    result3 = chat_rag.chat(
        "What did we talk about first?",
        use_rag=False
    )
    print(f"Answer: {result3['answer']}\n")
    
    # Clean up
    chat_rag.close()