# Import
import io
from typing import Dict
import pandas as pd
import numpy as np
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

    print(f">>>>>>> File saved to {local_file_name}")
    return local_file_name

def split_documents(document_main: str, pdf_file_path: str, pat_name: str, member_id: str):
    """Split document into LangChain Document objects with metadata."""
    pages = document_main.split('\n PAGE NUMBER')[1:]
    docs = [
        Document(
            page_content=f"PAGE NUMBER:-  {page.strip()} ",
            metadata={
                "source": f"Page number: {i+1}",
                "patient_name": pat_name,
                "member_id": member_id,
                "file_path": pdf_file_path
            }
        )
        for i, page in enumerate(pages)
    ]

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
def combine_queries_with_kg(
    patient_name: str,
    icd: str,
    icd_description: Dict[str, str]

) -> Dict[str, Dict[str, str]]:
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

def ocr_google_vision(img: Image.Image) -> str:
    """Perform OCR on a PIL image using Google Vision API."""
    # Convert PIL Image to bytes
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    image_data = img_byte_arr.getvalue()

    # Initialize Google Vision client
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_data)

    # Perform text detection
    response = client.text_detection(image=image)

    # Check for errors
    if response.error.message:
        raise Exception(f'Google Vision API Error: {response.error.message}')

    # Extract full text (first entry contains the full combined text)
    if response.text_annotations:
        return response.text_annotations[0].description.strip()
    else:
        return ""

def ocr_from_images_dict(
    images_dict: Dict[int, Image.Image], max_workers: int = 1
) -> str:
    """OCR all images from a dict of page_number: PIL.Image using Google Vision."""
    def process(page_number: int, image: Image.Image):
        text = ocr_google_vision(image)
        return (
            f"\n PAGE NUMBER:- {page_number}----------------------------------------\nDATA: {text}",
            page_number,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(lambda item: process(*item), images_dict.items()))

    results.sort(key=lambda x: x[1])
    final_texts = [r[0] for r in results]
    return "".join(final_texts), len(final_texts)


def detect_text_from_image(image_data):
    """
    Detects text in an image file using Google Vision API and returns it.
    
    Args:
        image_data (bytes): The image data in bytes format.
        
    Returns:
        str: The detected text.
    """
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_data)

    # Perform text detection
    response = client.text_detection(image=image)
    texts = response.text_annotations

    # Extract and return the detected text (first annotation contains full text)
    if texts:
        return texts[0].description
    else:
        return ""