import json
from ChunkCaptioner import ImageCaptioner
from pathlib import Path

sys_prompt = """
ROLE:
You are a technical documentation assistant generating captions for ORAN and 5G network diagrams.

TASK:
Generate a concise caption describing the image for technical documentation.

MANDATORY RULES:

1. ENTITY IDENTIFICATION
- Multiple boxes with identical labels = ONE entity at different time points
- Never say "first X" or "second X" - only "X"

2. ARROW DIRECTION
- CRITICAL: Trace arrows from tail to arrowhead
- "←" means flow FROM RIGHT TO LEFT
- "→" means flow FROM LEFT TO RIGHT
- Describe as "from X to Y"
- Two opposite arrows = two distinct messages (not bidirectional)
- Only call bidirectional if single arrow has arrowheads on both ends

3. MESSAGE SEQUENCING
- List ALL messages in top-to-bottom order
- Format: "Message Name" sent from [Source] to [Destination]
- Skip duplicate message interactions

4. CONTENT RESTRICTIONS
- Use ONLY visible information
- Use acronyms exactly as shown (don't expand)
- No assumptions or external knowledge
- Professional language only

EXAMPLE 1:
Input: Near-RT RIC and E2 Node sequence diagram
Output: The diagram shows interactions between a Near-RT RIC and an E2 Node:
1. "RIC SUBSCRIPTION REQUEST" from Near-RT RIC to E2 Node
2. "RIC SUBSCRIPTION FAILURE" from E2 Node to Near-RT RIC

EXAMPLE 2:
Input: Near-RT RIC and E2 Node single message
Output: The diagram shows: "RIC SUBSCRIPTION REQUEST" from E2 Node to Near-RT RIC

EXAMPLE 3:
Input: Heading/title only
Output: ""
 """

json_folder = "clean_chunks"  # Change this to your folder path
json_files = list(Path(json_folder).glob("*.json"))
print(f"Found {len(json_files)} JSON files")

blocktypes = ["Figure", "FigureGroup"] # if some are missing include "Pictures" as a blocktype

captioner = ImageCaptioner(system_prompt=sys_prompt)


for json_path in json_files:
    print(f"\nProcessing: {json_path.name}")
    try:
        with open(json_path, 'r') as file:
            data = json.load(file)
        for block in data:
            if block["block_type"] in blocktypes:
                description = captioner.generate_caption(base64_string=block["images"])
                block['description'] = description
                # show_image(img_b64)  # Display the image in the notebook
                # print(description)  # Output the generated caption
    
        with open(json_path, 'w') as file:
            json.dump(data, file, indent=4)
        print(f"  ✓ Updated {json_path.name}")
    except Exception as e:
        print(f"  ✗ Failed to process {json_path.name}: {e}")
        continue
print("Caption generation completed.")