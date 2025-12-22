import pymupdf


def extract_pdf_content(pdf_path):
    doc = pymupdf.open(pdf_path)

    text_collection = []
    image_collection = []

    for page_num, page in enumerate(doc):
        page_dict = page.get_text("dict")

        for block_id, block in enumerate(page_dict["blocks"]):
            if block["type"] == 0:
                text = " ".join(
                    span["text"]
                    for line in block["lines"]
                    for span in line["spans"]
                ).strip()

                if text:
                    text_collection.append({
                        "id": f"text_p{page_num}_b{block_id}",
                        "content": text,
                        "metadata": {
                            "page": page_num,
                            "bbox": block["bbox"],
                            "modality": "text"
                        }
                    })
        
        for img_id, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base = doc.extract_image(xref)

            image_collection.append({
                "id": f"image_p{page_num}_{img_id}",
                "image_bytes": base["image"],
                "format": base["ext"],
                "metadata": {
                    "page": page_num,
                    "xref": xref,
                    "modality": "image"
                }
            })
    return text_collection, image_collection


# if __name__ == "__main__":
#     texts, images = extract_pdf_content("document.pdf")
#     print(f"Extracted {len(texts)} text blocks and {len(images)} images.")