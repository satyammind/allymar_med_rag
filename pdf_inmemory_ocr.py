
# # from google.cloud import bigquery, storage
# # import os
# # os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/allymar_med_rag/allymarclinicalnotesapp-0c6d62f3197f.json"
# # def get_file(bucket_name: str, file_path: str, local_file_name: str) -> str:
# #     """Download a file from a GCS bucket and save it locally."""
# #     storage_client = storage.Client()
# #     blob = storage_client.bucket(bucket_name).blob(file_path)
# #     content = blob.download_as_bytes()

# #     with open(local_file_name, "wb") as f:
# #         f.write(content)

# #     print(f">>>>>>> File saved to {local_file_name}")
# #     return local_file_name

# # bucket = "dev-filestore-healthrecords"

# # file_name = "pdf"
# # pdf_path = get_file(bucket_name=bucket, file_path="LLM-Test/Amelia_Harris_Redacted_EDited.pdf", local_file_name=file_name.split('/')[-1])














# from pdf2image import convert_from_path
# from typing import Tuple
# from PIL import Image
# from concurrent.futures import ThreadPoolExecutor
# import gc  # for memory cleanup
# from paddleocr import PaddleOCR
# import numpy as np

# def ocr_paddle(img: Image.Image) -> str:
#     # Convert PIL Image to numpy array if necessary
#     if isinstance(img, Image.Image):
#         img = np.array(img)
#     ocr = PaddleOCR(use_angle_cls=True, lang="en")
#     result = ocr.ocr(img, cls=True)
#     return "\n".join([line[1][0] for res in result for line in res])
# def ocr_single_page(pdf_path: str, page_number: int, dpi: int = 300) -> Tuple[str, int]:
#     """Converts and OCRs a single PDF page."""
#     images = convert_from_path(pdf_path, dpi=dpi, first_page=page_number, last_page=page_number)
#     print('imagesimagesimagesimagesimagesimagesimagesimages')
#     image = images[0]
#     text = ocr_paddle(image)
#     del image
#     gc.collect()  # cleanup image from memory
#     return (
#         f"\n PAGE NUMBER:- {page_number}----------------------------------------\nDATA: {text}",
#         page_number,
#     )

# def ocr_from_pdf_streaming(pdf_path: str, dpi: int = 300, max_workers: int = 1) -> Tuple[str, int]:
#     """OCRs a PDF by processing one page at a time to reduce memory usage."""
#     # from pdf2image.pdfinfo import pdfinfo_from_path
#     from pdf2image import pdfinfo_from_path

#     num_pages = pdfinfo_from_path(pdf_path)["Pages"]

#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         results = list(executor.map(lambda p: ocr_single_page(pdf_path, p, dpi), range(1, num_pages + 1)))

#     results.sort(key=lambda x: x[1])
#     final_texts = [r[0] for r in results]
#     return "".join(final_texts), len(final_texts)


# pdf_path = "/home/mind/Downloads/Graph_doc.pdf"
# final_text, total_pages = ocr_from_pdf_streaming(pdf_path, dpi=300, max_workers=2)

# # Optionally save to a text file
# with open("output.txt", "w", encoding="utf-8") as f:
#     f.write(final_text)

# print(f"Processed {total_pages} pages.")












from pdf2image import convert_from_path
import base64
from io import BytesIO

# Parameters
pdf_path = "600_pages.pdf"
page_number = 1
dpi = 300
output_text_file = "image_base64.txt"

# Convert specific page to image
images = convert_from_path(pdf_path, dpi=dpi, first_page=page_number, last_page=page_number)

# Save image content to text file in base64 format
with open(output_text_file, "w", encoding="utf-8") as f:
    for image in images:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        f.write(img_base64)
