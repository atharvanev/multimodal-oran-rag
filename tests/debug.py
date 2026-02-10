import requests

# Delete
requests.delete("http://172.17.0.2:8080/v1/schema/Grounded_nomic_full")

# Recreate with correct skip settings via REST API
schema = {
    "class": "Grounded_nomic_full",
    "vectorConfig": {
        "default": {
            "vectorIndexType": "hnsw",
            "vectorizer": {
                "text2vec-ollama": {
                    "apiEndpoint": "http://172.17.0.4:11434",
                    "model": "nomic-embed-text",
                    "vectorizeClassName": True
                }
            }
        }
    },
    "moduleConfig": {
        "generative-ollama": {
            "apiEndpoint": "http://172.17.0.4:11434",
            "model": "llama3.2"
        }
    },
    "properties": [
        {"name": "type",        "dataType": ["text"],  "moduleConfig": {"text2vec-ollama": {"skip": True,  "vectorizePropertyName": False}}},
        {"name": "page",        "dataType": ["int"],   "moduleConfig": {"text2vec-ollama": {"skip": True,  "vectorizePropertyName": False}}},
        {"name": "description", "dataType": ["text"],  "moduleConfig": {"text2vec-ollama": {"skip": False, "vectorizePropertyName": False}}},
        {"name": "text",        "dataType": ["text"],  "moduleConfig": {"text2vec-ollama": {"skip": False, "vectorizePropertyName": False}}},
        {"name": "trace",       "dataType": ["text"],  "moduleConfig": {"text2vec-ollama": {"skip": False, "vectorizePropertyName": False}}},
        {"name": "filename",    "dataType": ["text"],  "moduleConfig": {"text2vec-ollama": {"skip": False, "vectorizePropertyName": False}}},
        {"name": "image",       "dataType": ["blob"],  "moduleConfig": {"text2vec-ollama": {"skip": True,  "vectorizePropertyName": False}}}
    ]
}

r = requests.post("http://172.17.0.2:8080/v1/schema", json=schema)
print(r.status_code, r.text)

r = requests.get("http://172.17.0.2:8080/v1/schema/Grounded_nomic_full")
for p in r.json()["properties"]:
    print(p["name"], p["moduleConfig"])