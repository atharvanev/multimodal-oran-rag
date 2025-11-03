# %%
import json
#from ChunkCaptioner import ImageCaptioner
from pathlib import Path
import html2text
import re

# %%
# json_path = Path("Output/38473-h20/38473-h20.json")
# new_path = json_path.with_name(json_path.stem + "_cleaned" + json_path.suffix)

# %%
def cleanJSON(path,output_path=None):
    json_path = Path(path)
    # Determine new_path based on output_path (dir or file). Create dirs if needed.
    if output_path:
        out = Path(output_path)
        # treat as directory when no suffix or path exists as dir
        if out.suffix == "" or (out.exists() and out.is_dir()):
            out.mkdir(parents=True, exist_ok=True)
            new_path = out / (json_path.stem + "_cleaned" + json_path.suffix)
        else:
            out.parent.mkdir(parents=True, exist_ok=True)
            new_path = out
    else:
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
    usless_keys = ["polygon", "bbox"]

    data = data['blocks']

    datalookup = {x["id"]: x for x in data}


    for block in data:
        block['description'] = ""

        block["images"] = list(block["images"].values())[0] if block["images"] else "" #ASSUMES THAT ONLY 1 IMAGE PER BLOCK IF ERRROR LATER THIS COULD BE THE REASON
        
        for key in usless_keys: #removes usless keys
            if key in block:
                del block[key]

        if block["block_type"] == "Table" or block["block_type"] == "TableofContents":
            block["text"] = html2text.html2text(block["html"])
        else:
            text = h.handle(block["html"])           # First: convert HTML to text
            text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)  # Then: clean it
            block["text"] = text.strip()    
            
        del block["html"]

        block["page"] = int(Path(block["id"]).parts[2])+1  #gets page number from id
        block["filename"] = json_path.stem  #gets file name from file path

    
    for i, block in enumerate(data):
        tree = []
        for _,id in block["section_hierarchy"].items():
            tree.append(datalookup[id]["text"])
        if block["block_type"] == "Figure":
            block["text"] = data[i+1]["text"] if i+1 < len(data) else ""
            tree.append(block["text"])
        elif block["block_type"] == "FigureGroup":
            tree.append(block["text"]) 

        
        block["trace"]= " --> ".join(tree)
        del block["section_hierarchy"]
            
    with open(new_path, 'w') as file:
        json.dump(data, file, indent=4)

    return new_path


#cleanJSON("Output/38473-h20/38473-h20.json")