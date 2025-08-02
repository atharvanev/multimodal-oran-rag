import fitz
import os
from PIL import Image
from docx2pdf import convert
from IPython.display import Image, display
from pathlib import Path


def get_Text_and_Images(file_path="files/O-RAN.WG3.TS.E2AP-R004-v07.00 (1).pdf"):
    #file_path = "files/O-RAN.WG3.TS.E2AP-R004-v07.00 (1).pdf"
    folder_name = os.path.splitext(os.path.basename(file_path))[0]

    # Create the full path: ./Images/<folder_name>
    output_dir = Path("Images") / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    #open pdf
    pdf_file = fitz.open(file_path)

    #define pdf length
    page_nums = len(pdf_file)


    Images_list= []
    Text_list = []
    for page_num in range(page_nums):
        page = pdf_file[page_num]
        page_dict = page.get_text("dict")
        blocks = page_dict["blocks"]
        
        textblocks = [b for b in blocks if b["type"] == 0]
        imageblocks = [b for b in blocks if b["type"] == 1]


        for t in textblocks:
            t['page'] = int(page_num)+1
        Text_list.extend(textblocks)
        
        with open("files/logo.jpeg","rb") as logo:
            logo_bytes = logo.read()

        page_list = [
        image for image in imageblocks
        if image["image"] and image["image"] != logo_bytes and len(image["image"]) > 700
        ]
        if page_list:
            for p in page_list:
                p['page'] = int(page_num)+1
            Images_list.extend(page_list)
        for i,img in enumerate(page_list, start =1):
            bytes = img["image"]
            ext = img["ext"]
            name = F"page{page_num+1}_img{i}.{ext}"

            with open(os.path.join(output_dir,name),'wb') as image_file:
                image_file.write(bytes)
                image_file.close()
    return Images_list, Text_list

#isolate information only we want
def clean_images(Images_list):
    cleaned_images = []
    for img in Images_list:
        cleaned_images.append({"img":img['image'],"bbox":img['bbox'],"page":img['page']})
    return cleaned_images

def clean_text(Text_list):
    cleaned_text = []
    for i in range(len(Text_list)):
        Textblock = []
        bbox = None
        page = None
        for line in Text_list[i]['lines']:
            for span in line['spans']:
                Textblock.append(span['text'])
            bbox = Text_list[i]['bbox']
            page = Text_list[i]['page']
        if "".join(Textblock).strip() != "":
            cleaned_text.append({"text":"".join(Textblock),"bbox":bbox,"page":page})
    return cleaned_text

def find_caption(image,cleaned_text):
    best_text = None
    min_distance = float('inf')
    for text in cleaned_text:
        if text["page"] == image["page"]:
            if text["bbox"][1] > image["bbox"][3]:
                vertical_distance = image["bbox"][1] - text["bbox"][3]
                if round(abs(vertical_distance)) < min_distance:
                    min_distance = abs(vertical_distance)
                    best_text = text
    return best_text['text']

def add_captions(cleaned_image,cleaned_text):
    captioned_photos = cleaned_image
    for img in captioned_photos:
        img['caption']= find_caption(img,cleaned_text)
    return captioned_photos

# Images_list,Text_list = get_Text_and_Images()
# cleaned_images = clean_images(Images_list)
# cleaned_text = clean_text(Text_list)
# captioned_photos = add_captions(cleaned_images,cleaned_text)

# for img in captioned_photos:
#     display(Image(data=img['img']))
#     print(img['caption'],f"on page {img['page']}")
