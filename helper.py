# Import
import io
import json
from typing import Dict
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from google.cloud import bigquery, storage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from google.cloud import vision

# BigQuery Utilities
def df_from_bigquery(table: str = None, custom_sql: str = None) -> pd.DataFrame:
    """
    Create a dataframe from a BigQuery table or custom SQL query.
    """
    client = bigquery.Client()
    query = custom_sql if custom_sql else f"SELECT * FROM `{table}`"
    return client.query(query).to_dataframe()


def get_file(bucket_name: str, file_path: str, local_file_name: str) -> str:
    """Download a file from a GCS bucket and save it locally."""
    storage_client = storage.Client()
    blob = storage_client.bucket(bucket_name).blob(file_path)
    content = blob.download_as_bytes()
    with open(local_file_name, "wb") as f:
        f.write(content)
    return local_file_name


def save(bucket_name, data, destination_blob_name):
    """Upload data directly to a specified location in a Google Cloud Storage bucket.
    data = json.dumps(data)
    or pass data directly if string
    Args:
        bucket_name (str): Name of the bucket.
        data (str or dict): Data to be saved. Can be a string or a dictionary (for JSON).
        destination_blob_name (str): Blob name in the bucket where data will be saved, including path.
        data_type (str): Type of the data ('text' or 'json'). Defaults to 'text'.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_string(data)


def split_documents(document_main: str,pdf_file_path: str,pat_name: str,member_id: str):
    """Split document into LangChain Document objects with metadata."""
    pages = document_main.split('\n PAGE NUMBER')[1:]

    docs = []
    for i, page in enumerate(pages):
        page_number = i + 1

        doc = Document(
            page_content=f"PAGE NUMBER:-  {page.strip()} ",
            metadata={
                "source": f"Page number: {page_number}",
                "patient_name": pat_name,
                "member_id": member_id,
                "file_path": pdf_file_path,
            }
        )
        docs.append(doc)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
    )

    return text_splitter.split_documents(docs)


# TAMPER Query Generation
def generate_tamper_queries(icd: str, patient_name: str) -> Dict[str, str]:
    """Generate standard TAMPER-style clinical queries."""
    return {
        "Treatment": f"What specific treatments are documented for {patient_name} with {icd}?",
        "Assessment": f"What assessments have been performed for {patient_name}'s {icd}?",
        "Monitoring": f"How is {patient_name}'s {icd} being monitored?",
        "Plan": f"What is the care plan for {patient_name}'s {icd}?",
        "Evaluation": f"How is {patient_name}'s response to {icd} treatment being evaluated?",
        "Referral": f"What referrals have been made for {patient_name}'s {icd}?",
        "Full_Knowledge": f"Provide all relevant information about {patient_name}'s {icd}."
    }

# Query-Knowledgebase Combiner
def combine_queries_with_kg(patient_name: str, icd: str, icd_description: Dict[str, str]) -> Dict[str, Dict[str, str]]:
    """
    Combine TAMPER queries with ICD knowledge graph snippets.
    """
    queries = generate_tamper_queries(icd, patient_name)
    combined = {}

    for section, query in queries.items():
        if section != "Full_Knowledge":
            kg_text = icd_description.get(section[0], "")  # Use initial letter: T, A, M, etc.
        else:
            kg_text = " ".join(icd_description.values())

        combined[section] = {
            "query": query,
            "knowledge_base": kg_text
        }
    return combined


def ocr_google_vision(img: Image.Image) -> Dict:
    """Perform OCR on a PIL image using Google Vision DOCUMENT_TEXT_DETECTION API and return structured data."""

    # Convert PIL Image to bytes
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    image_data = img_byte_arr.getvalue()

    # Initialize Google Vision client
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_data)

    # Use DOCUMENT_TEXT_DETECTION for full annotation
    response = client.document_text_detection(image=image)
    if response.error.message:
        raise Exception(f'Google Vision API Error: {response.error.message}')

    # Build structured response_dict following required format
    response_dict = {
        "responses": [
            {
                "full_text_annotation": {
                    "text": response.full_text_annotation.text,
                    "pages": [
                        {
                            "blocks": [
                                {
                                    "confidence": block.confidence,
                                    "paragraphs": [
                                        {
                                            "confidence": par.confidence,
                                            "words": [
                                                {
                                                    "bounding_box": [
                                                        (vertex.x, vertex.y)
                                                        for vertex in word.bounding_box.vertices
                                                    ],
                                                    "confidence": word.confidence,
                                                    "symbols": [
                                                        {
                                                            "text": symbol.text,
                                                            "confidence": symbol.confidence,
                                                        }
                                                        for symbol in word.symbols
                                                    ],
                                                }
                                                for word in par.words
                                            ],
                                        }
                                        for par in block.paragraphs
                                    ],
                                }
                                for block in page.blocks
                            ]
                        }
                        for page in response.full_text_annotation.pages
                    ],
                }
            }
        ]
    }
    return response_dict


def ocr_from_images_dict(bucket_name: str, destination_blob_name: str,images_dict: Dict[int, Image.Image], max_workers: int = 1) -> str:
    """OCR all images from a dict of page_number: PIL.Image using Google Vision."""
    annotations = {}
    results = []
    
    def process(page_number: int, image: Image.Image):
        ann = ocr_google_vision(image)
        # Return the annotation with the page number so we can update annotations outside
        try:
            text = ann["responses"][0]["full_text_annotation"]["text"]
        except (KeyError, IndexError):
            text = ""
        return f"\n PAGE NUMBER:- {page_number}----------------------------------------\nDATA: {text}", page_number, ann
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Collect all results including the annotations
        process_results = list(executor.map(lambda item: process(*item), images_dict.items()))

    print("OCR completed successfully.")
    
    # Update annotations dictionary and prepare text results
    for result, page_number, annotation in process_results:
        annotations[page_number] = annotation
        results.append((result, page_number))

    # save to gcs
    print("Proceeding to save the annotations in gcs...")
    save(bucket_name=bucket_name, destination_blob_name=destination_blob_name, data=json.dumps(annotations))
    
    # Sort results by page number and join
    results.sort(key=lambda x: x[1])
    return "".join(r[0] for r in results)


# calculate the size of all images converted from pdf
def calculate_total_image_size_gb(image_dict):
    total_bytes = 0
    for page_num, img in image_dict.items():
        width, height = img.size
        channels = len(img.getbands()) 
        total_bytes += width * height * channels

    total_gb = total_bytes / (1024 ** 3)
    return round(total_gb, 2)

