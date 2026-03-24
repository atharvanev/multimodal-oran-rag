import weaviate
from collections import Counter

client = weaviate.connect_to_local(
    host="172.17.0.5",
    port=8080,
    grpc_port=50051,
)

collection = client.collections.get("Grounded_nomic_full")

block_types = Counter()

for obj in collection.iterator(return_properties=["type"]):
    bt = obj.properties.get("type", "None/Missing")
    block_types[bt] += 1

client.close()

print("\n=== Block Types ===")
for bt, count in block_types.most_common():
    print(f"  {bt}: {count}")
print(f"\nTotal objects: {sum(block_types.values())}")