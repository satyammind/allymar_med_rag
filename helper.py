# Import
import re
from typing import Dict
import pandas as pd
from google.cloud import bigquery, storage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from google import genai
import os
import vertexai

PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION")

vertexai.init(location= REGION , project=PROJECT_ID)


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "allymarclinicalnotesapp-0c6d62f3197f.json"

client = genai.Client(vertexai=True, project=PROJECT_ID, location=REGION)

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

def preprocess_page(page_text):
    """Split page into logical sections using headings or blocks."""
    return re.split(r"\n(?=\s*[A-Z ]{3,20}:)", page_text.strip())

def split_documents(document_main: str, pdf_file_path: str, pat_name: str, member_id: str):
    """Split document into LangChain Document objects with metadata."""
    if isinstance(document_main, list):
        document_main = "\n".join(document_main)
    
    pages = document_main.split('\n PAGE NUMBER')[1:]
    docs = []

    for i, page in enumerate(pages):
        page_number = i + 1
        sections = preprocess_page(page)

        for section in sections:
            doc = Document(
                page_content=section.strip(),
                metadata={
                    "source": f"Page number: {page_number}",
                    "patient_name": pat_name,
                    "member_id": member_id,
                    "file_path": pdf_file_path,
                }
            )
            docs.append(doc)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", ";"]
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
            kg_text = icd_description.get(section[0], "")  
        else:
            kg_text = " ".join(icd_description.values())

        combined[section] = {
            "query": query,
            "knowledge_base": kg_text
        }
    return combined
