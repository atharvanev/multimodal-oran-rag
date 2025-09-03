# %%
import json
from ChunkCaptioner import ImageCaptioner
from pathlib import Path
import html2text
import re

# %%
json_path = Path("Output/38473-h20/38473-h20.json")
new_path = json_path.with_name(json_path.stem + "_cleaned" + json_path.suffix)

# %%
def cleanJSON(path):
    json_path = Path(path)
    new_path = json_path.with_name(json_path.stem + "_cleaned" + json_path.suffix)
    h = html2text.HTML2Text()

    h.ignore_links = True
    h.ignore_images = True
    h.body_width = 0
    h.ignore_emphasis = True
    h.skip_internal_links = True
    h.single_line_break = True
    h.mark_code = False
    h.protect_links = False
    h.ignore_tables = True
    h.escape_snob = True
    h.inline_links = True
    h.default_image_alt = ""  # Remove alt text
    h.use_automatic_links = False

    with open(json_path, 'r') as file:
        data = json.load(file)
    usless_keys = ["polygon", "bbox",]

    data = data['blocks']

    datalookup = {x["id"]: x for x in data}


    for block in data:
        for key in usless_keys:
            if key in block:
                del block[key]
        if block["block_type"] == "Table" or block["block_type"] == "TableofContents":
            block["Text"] = html2text.html2text(block["html"])
        else:
            text = h.handle(block["html"])           # First: convert HTML to text
            text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)  # Then: clean it
            block["Text"] = text.strip()    
            
        del block["html"]
        block["page"] = int(Path(block["id"]).parts[2])+1

    for i, block in enumerate(data):
        tree = []
        for _,id in block["section_hierarchy"].items():
            tree.append(datalookup[id]["Text"])
        if block["block_type"] == "Figure":
            block["Text"] = data[i+1]["Text"] if i+1 < len(data) else ""
            tree.append(block["Text"])
        elif block["block_type"] == "FigureGroup":
            tree.append(block["Text"]) 

        
        block["Trace"]= " --> ".join(tree)
        del block["section_hierarchy"]
            
    with open(new_path, 'w') as file:
        json.dump(data, file, indent=4)

    return new_path


cleanJSON("Output/38473-h20/38473-h20.json")