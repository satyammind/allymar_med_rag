# Import
from typing import Dict
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
from pdf2image import convert_from_path
from paddleocr import PaddleOCR

from google.cloud import bigquery, storage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# BigQuery Utilities
def df_from_bigquery(table: str = None, custom_sql: str = None) -> pd.DataFrame:
    """
    Create a dataframe from a BigQuery table or custom SQL query.
    """
    client = bigquery.Client()
    query = custom_sql if custom_sql else f"SELECT * FROM `{table}`"
    return client.query(query).to_dataframe()

# Get Patient Name
def get_patient_name_by_member_id(table: str, member_id: str) -> str:
    client = bigquery.Client()
    query = f"""
        SELECT patient_name
        FROM `{table}`
        WHERE member_id = '{member_id}'
        LIMIT 1
    """
    df = client.query(query).to_dataframe()
    return df['patient_name'].iloc[0] if not df.empty else None

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
    table: str,
    member_id: str,
    icd: str,
    icd_description: Dict[str, str]
) -> Dict[str, Dict[str, str]]:
    """
    Combine TAMPER queries with ICD knowledge graph snippets.
    """
    patient_name = get_patient_name_by_member_id(table, member_id)
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


def ocr_paddle(img: Image.Image) -> str:
    # Convert PIL Image to numpy array if necessary
    if isinstance(img, Image.Image):
        img = np.array(img)
    ocr = PaddleOCR(use_angle_cls=True, lang="en")
    result = ocr.ocr(img, cls=True)
    return "\n".join([line[1][0] for res in result for line in res])

def ocr_from_images_dict(
    images_dict: Dict[int, Image.Image], max_workers: int = 1
) -> str:
    """OCR all images from a dict of page_number: PIL.Image"""
    def process(page_number: int, image: Image.Image):
        text = ocr_paddle(image)
        return (
            f"\n PAGE NUMBER:- {page_number}----------------------------------------\nDATA: {text}",
            page_number,
        )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(lambda item: process(*item), images_dict.items()))

    results.sort(key=lambda x: x[1])
    final_texts = [r[0] for r in results]
    return "".join(final_texts), len(final_texts)
