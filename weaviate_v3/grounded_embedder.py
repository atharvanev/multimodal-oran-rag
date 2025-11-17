import weaviate
from weaviate.classes.config  import Configure,Property, DataType
import requests, json
import base64
from pathlib import Path
import os


client = weaviate.connect_to_local(
    host="172.17.0.5",  
    port=8080,
    grpc_port=50051,
)

#Only need to run once to test create the collection

collections = client.collections.delete("Grounded_nomic_full")
print(collections)


questions = client.collections.create(
    name="Grounded_nomic_full",
    vector_config=Configure.Vectors.text2vec_ollama(  # Configure the Ollama embedding integration
        api_endpoint="http://172.17.0.3:11434",  # If using Docker you might need: http://host.docker.internal:11434
        model="nomic-embed-text:latest",  # The model to use
    ),
    generative_config=Configure.Generative.ollama(  # Configure the Ollama generative integration
        api_endpoint="http://172.17.0.3:11434",  # If using Docker you might need: http://host.docker.internal:11434
        model="llama3.2",  # The model to use
    ),
    properties=[
        Property(
            name="type",
            data_type=DataType.TEXT,
            skip_vectorization=False
        ),
        Property(
            name="page",
            data_type=DataType.INT,
            skip_vectorization=False
        ),
        Property(
            name="description",
            data_type=DataType.TEXT,
            skip_vectorization=False
        ),
        Property(
            name="text",
            data_type=DataType.TEXT,
            skip_vectorization=False
        ),
        Property(
            name="trace",
            data_type=DataType.TEXT,
            skip_vectorization=False
        ),
        Property(
            name="filename",
            data_type=DataType.TEXT,
            skip_vectorization=False
        ),
        Property(
            name="image",
            data_type=DataType.BLOB,
            skip_vectorization=True,  # Image won't be embedded
            index_null_state=True
        ),
    ]
)

questions = client.collections.use("Grounded_nomic_full")

for filename in os.listdir("multimodal-oran-rag/clean_chunks"):
    if filename.endswith(".json"):
        filepath = os.path.join("../clean_chunks", filename)
            
    with open(filepath, 'r') as f:
        data = json.load(f)

    with questions.batch.fixed_size(batch_size=200) as batch:
        for d in data:
            properties = {
                    "type": d["block_type"],
                    "page": d["page"],
                    "description": d["description"],
                    "text": d["text"],
                    "trace": d["trace"],
                    "filename": d["filename"],
                #    "Image": list(d["images"].values())[0] if d["images"] else None,
                }

            # Handle image properly
            if d["images"]:
                properties["image"] = d["images"]
        
            batch.add_object(properties)
            
            if batch.number_errors > 10:
                print("Batch import stopped due to excessive errors.")
                break

    failed_objects = questions.batch.failed_objects
    if failed_objects:
        print(f"Number of failed imports: {len(failed_objects)}")
        print(f"First failed object: {failed_objects[0]}")

client.close()  # Free up resources