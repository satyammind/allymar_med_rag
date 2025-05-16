# allymar/MemberRag.py

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date
import json
import os
import shutil
from typing import List, Any, Dict, Optional, Union
from retry import retry
from pdf2image import convert_from_path
import vertexai
from helper import calculate_total_image_size_gb, df_from_bigquery, get_file, split_documents, combine_queries_with_kg, ocr_from_images_dict
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
from PIL import Image
from langchain_google_community import BigQueryVectorStore
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from dotenv import load_dotenv
from PyPDF2 import PdfReader


load_dotenv(verbose=True)

# Set up Google Cloud credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="allymarclinicalnotesapp-0c6d62f3197f.json"
PROJECT_ID = os.getenv("PROJECT_ID")
DATASET = os.getenv("DATASET")
TABLE = os.getenv("TABLE")
REGION = os.getenv("REGION")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")

# from langchain_google_genai import GoogleGenerativeAIEmbeddings
os.environ["GOOGLE_GENAI_USE_VERTEXAI"]="TRUE"

# embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")
EMBEDDING_MODEL = VertexAIEmbeddings(model_name=EMBEDDING_MODEL_NAME, project=PROJECT_ID, location=REGION)

vertexai.init(location= REGION , project=PROJECT_ID)
llm = VertexAI(model_name="gemini-2.0-flash")

# DEFAULT_K = os.getenv("DEFAULT_K")
# DEFAULT_THRESHOLD = os.getenv("DEFAULT_THRESHOLD")

DEFAULT_K = int(os.getenv("DEFAULT_K", 5))
DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD", 0.8))


## Set file-path , table name, member_id and bucket name
table = f"{PROJECT_ID}.{DATASET}.{TABLE}"
bucket = os.getenv("BUCKET_NAME")

metadata={"member_id": "37149e94-216e-474b-af8b-3227b73da082"}
patient_name = "John Doe"


# Initialize the bigquery for data ingestion and retrieval
bq_store = BigQueryVectorStore(
    project_id=PROJECT_ID,
    dataset_name=DATASET,
    table_name=TABLE,
    location=REGION,
    embedding=EMBEDDING_MODEL,
)


@dataclass(slots=True)
class MemberRAG:
    compressor = FlashrankRerank()
    member_id: str = field(default=metadata.get("member_id"))
    table: str = field(default=table)
    bucket: str = field(default=bucket)
    file_name: str = field(default="")
    _documents: Optional[Any] = None

    def get_documents(self):
        """Get documents from BigQuery and create a index."""
        if self._documents is None:
            sql = (
                f"SELECT doc_id, content, embedding, source, chunk, file_path, patient_name "
                f"FROM `{self.table}` "
                f"WHERE member_id = '{self.member_id}'"
            )
            df = df_from_bigquery(custom_sql=sql)
            documents = [
                Document(
                    page_content=row["content"],
                    metadata={
                        "doc_id": row["doc_id"],
                        "source": row["source"],
                        "chunk": row["chunk"],
                        "file_path": row["file_path"],
                        "patient_name": row["patient_name"],
                        "member_id": self.member_id
                    }
                )
                for _, row in df.iterrows()
            ] 
            self._documents = documents
            print(f">>>>>>> Documents initialized for member_id: {self.member_id}")

    def pdf_to_images(self, file_name: str, dpi: int = 300) -> Dict[int, Image.Image]:
        """Convert a PDF into a dict of images: {page_number: Image}"""
        # get file path for OCR
        bq_file_name=f"LLM-Test/{self.member_id}/{file_name}"
        pdf_path = get_file(bucket_name=bucket, file_path=bq_file_name, local_file_name=file_name.split('/')[-1])
        # images = convert_from_path(pdf_path, dpi=dpi)
        num_pages = len(PdfReader(pdf_path).pages)
        
        images_dict = {}

        for page_number in range(1, num_pages + 1):
            images = convert_from_path(
                pdf_path,
                dpi=dpi,
                first_page=page_number,
                last_page=page_number
            )
            if images:
                images_dict[page_number] = images[0]

        # remove the downloaded file
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        else:
            print(f"The file {pdf_path} does not exist")

        print("Pdf's are converted to images successfully.")
        total_size_gb = calculate_total_image_size_gb(images_dict)
        print(f"Total size of images: {total_size_gb} GB")

        return images_dict


    # @retry(max_retries=5, backoff_factor=.5, verbose=True)
    def add_documents_to_index(self, images: dict, file_name: str, metadata: dict= {}) -> bool:
        """Add documents to the index."""
        try:
            bq_file_name=f"LLM-Test/{self.member_id}/{file_name}"
            gcs_file_path = f"gs://{self.bucket}/{bq_file_name}"
            destination_file_name = f"LLM-Test/{self.member_id}/annotations/{file_name}"

            print("Performing OCR on retrieved images...")
            document_main= ocr_from_images_dict(images_dict=images, bucket_name=bucket, destination_blob_name=destination_file_name)

            doc_splits = split_documents(document_main=document_main, pdf_file_path=gcs_file_path, pat_name=patient_name, member_id=self.member_id)

            # Add chunk number to metadata
            for idx, split in enumerate(doc_splits):
                split.metadata["chunk"] = idx

            # Data Ingestion to BigQuery
            print("Proceeding with Ingesting data to Bigquery...")
            bq_store.add_documents(doc_splits)

            print("Indexing success:", True)

            return True

        except Exception as e:
            # Raise exception with context
            raise RuntimeError(f"Failed to verify/create index for member_id={metadata.get('member_id')}: {e}")


    def get_relevant_docs_and_metadata(self, question: str, k: int = DEFAULT_K, threshold: float = DEFAULT_THRESHOLD) -> Union[List[Dict[str, Any]], Exception]:
        """
        Retrieves relevant Document objects and caches them in _documents.
        Returns a list of dicts with text and metadata.
        """
        try:
            hits = bq_store.similarity_search_with_score(
                query=question,
                k=k,
                filter={"member_id": self.member_id},
            )
            docs: List[Document] = []
            result: List[Dict[str, Any]] = []
            for doc, score in hits:
                if score >= threshold:
                    docs.append(doc)
                    result.append({
                        "text": doc.page_content,
                        "metadata": doc.metadata,
                    })
            # Cache Document objects for retriever
            self._documents = docs
            return result
        except Exception as e:
            return RuntimeError(
                f"Error retrieving relevant documents for member_id={self.member_id}: {e}"
            )


    def query(self, question: str, k: int = DEFAULT_K, threshold: float = DEFAULT_THRESHOLD) -> Union[str, Exception]:
        """Query the RAG model with a question and return the answer"""
        try:
            base_retriever = bq_store.as_retriever(k=k, threshold=threshold)
            comp_retriever = ContextualCompressionRetriever(
                base_compressor=self.compressor,
                base_retriever=base_retriever,
            )
            ensemble = EnsembleRetriever(
                retrievers=[base_retriever, comp_retriever],
                weights=[0.3, 0.7],
            )
            rag_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=ensemble,
            )
            return rag_chain.run(question)
        except Exception as e:
            return RuntimeError(f"RetrievalQA failed: {e}")
        
    
    def get_is_valid_suspect(self, icd: str, icd_description: dict) -> str | Exception:
        """Determine if condition is suspected today for a given ICD, using retrieved clinical evidence."""
        try:
            seen_ids = set()
            full_knowledge_seen_ids = set()
            response: dict[str, list] = {}

            # Step 1: Generate knowledge graph-enhanced queries
            combined_queries = combine_queries_with_kg(
                patient_name=patient_name,
                icd=icd,
                icd_description=icd_description
            )

            # Step 2: Retrieve documents and deduplicate by doc_id or content hash
            for section, data in combined_queries.items():
                docs = self.get_relevant_docs_and_metadata(
                    question=data["query"],
                    k=DEFAULT_K,
                    threshold=DEFAULT_THRESHOLD
                )
                unique_docs = []
                for doc in docs:
                    doc_id = doc['metadata'].get('doc_id') or hash(doc["text"])
                    if section == "Full_Knowledge":
                        if doc_id not in full_knowledge_seen_ids:
                            full_knowledge_seen_ids.add(doc_id)
                            unique_docs.append(doc)
                    else:
                        if doc_id not in seen_ids:
                            seen_ids.add(doc_id)
                            unique_docs.append(doc)

                response[section] = unique_docs

            # Step 3: Collect all deduplicated documents
            all_docs = {
                doc['metadata'].get('doc_id') or hash(doc["text"]): doc
                for section_docs in response.values()
                for doc in section_docs
            }

            today_str = date.today().strftime("%B %d, %Y")
          
            question = (
                f"Today is {today_str}.\n\n"
                f"Here is the information for the patient list data along with date information.\n\n"
                f"{all_docs}\n\n"
                f"Would you suspect the condition '{icd_description.get('description', icd)}' today?"
            )
            
            # Step 4: Query the LLM with the combined data
            llm_response = llm.invoke(question)
            return llm_response
        
        except Exception as e:
            raise RuntimeError(f"Error validating suspect for member_id={self.member_id}: {e}")



"==========================================================================================================================================="

# Example usage
result = MemberRAG(member_id=metadata.get("member_id"))

print("------------------------------------------------------------------------------------------------------------------------")
question1 = "What is the treatment plan for this patient?"
question2 = "What is the assessment for this patient?"
question3 = "What is the monitoring plan for this patient?"
question4 = "What is the evaluation plan for this patient?"
question5 = "What is the referral plan for this patient?"
question6 = "What is the full knowledge for this patient?"

print("------------------------------------------------------------------------------------------------------------------------")
# # Call the query method
# response1 = result.query(question=question1)
# print("Query response:11111111111111111111111111111111111111111111", response1)

# response2 = result.query(question=question2)
# print("Query response:22222222222222222222222222222222222222222222", response2)

# response3 = result.query(question=question3)
# print("Query response:3333333333333333333333333333333333333333333333333", response3)
# response4 = result.query(question=question4)
# print("Query response:4444444444444444444444444444444444444444444444444444", response4)
# response5 = result.query(question=question5)
# print("Query response:55555555555555555555555555555555555555555555555", response5)
# response6 = result.query(question=question6)    
# print("Query response:6666666666666666666666666666666666666666666666666", response6)

print("------------------------------------------------------------------------------------------------------------------------")
# Call the get_relevant_docs_and_metadata method
# data1 = result.get_relevant_docs_and_metadata(question=question1, k=DEFAULT_K, threshold=DEFAULT_THRESHOLD)
# print("relavent_docs===============================================================================:", data1)

# data2 = result.get_relevant_docs_and_metadata(question=question2, k=DEFAULT_K, threshold=DEFAULT_THRESHOLD)
# print("relavent_docs===============================================================================:", data2)   
# data3 = result.get_relevant_docs_and_metadata(question=question3, k=DEFAULT_K, threshold=DEFAULT_THRESHOLD)
# print("relavent_docs===============================================================================:", data3)
# data4 = result.get_relevant_docs_and_metadata(question=question4, k=DEFAULT_K, threshold=DEFAULT_THRESHOLD)
# print("relavent_docs===============================================================================:", data4)
# data5 = result.get_relevant_docs_and_metadata(question=question5, k=DEFAULT_K, threshold=DEFAULT_THRESHOLD)
# print("relavent_docs===============================================================================:", data5)
# data6 = result.get_relevant_docs_and_metadata(question=question6, k=DEFAULT_K, threshold=DEFAULT_THRESHOLD)
# print("relavent_docs===============================================================================:", data6)

print("------------------------------------------------------------------------------------------------------------------------")
# Call the pdf_to_images method
images = result.pdf_to_images(file_name="DEV_CCL summary (1).pdf")
# document_main , all_annonations = ocr_from_images_dict(images_dict=images)
# print("OCR text from images:" )

print("------------------------------------------------------------------------------------------------------------------------")
# call the add_documents_to_index method
success = result.add_documents_to_index(images=images, file_name="DEV_CCL summary (1).pdf", metadata=metadata)

print("------------------------------------------------------------------------------------------------------------------------")
icd_description =  {
        "T": """**T — Treatment** :Documentation for familial hypophosphatemia (FH) typically reflects treatment with **oral phosphate salts** (e.g., 20–80 mg/kg/day elemental phosphorus split into 4–6 daily doses) and **active vitamin D analogs** like calcitriol (0.25–0.75 µg/day) to counteract renal phosphate wasting and impaired mineralization[^1][^3]. Since 2018, **burosumab** (Crysvita), a monoclonal antibody targeting FGF23, is administered subcutaneously every 2–4 weeks for X-linked hypophosphatemia (XLH)[^1][^7]. Documentation often specifies medication dosages, frequency, and adjustments based on lab monitoring or adverse effects (e.g., hypercalciuria, nephrocalcinosis)[^3]. Supportive measures, such as dental sealants to prevent abscesses, may also be noted[^1]. Guideline-concordant care emphasizes individualized regimens to balance biochemical correction with avoidance of complications[^3][^7].""",
        "A": """**A — Assessment**: Clinical documentation typically includes **hypophosphatemia** (serum phosphate below age-adjusted norms), **elevated FGF23**, and **normal calcium/25-hydroxyvitamin D levels**, alongside **renal phosphate wasting** (elevated urinary phosphate or reduced TmP/GFR)[^1][^3][^7]. Radiographs may show **rickets** (children) or **osteomalacia** (adults), such as metaphyseal fraying or pseudofractures[^3][^7]. Genetic testing confirming *PHEX* (XLH) or *FGF23* (ADHR) variants is increasingly documented to differentiate FH from acquired causes[^1][^7]. Notes often exclude nutritional deficiencies or secondary causes (e.g., tumor-induced osteomalacia) and may reference family history of skeletal abnormalities[^3][^7].""",
        "M": """**M — Monitoring** :Longitudinal documentation includes **serial serum phosphate**, **alkaline phosphatase (ALP)**, **urinary calcium/creatinine ratios**, and **renal function tests** every 3–6 months to gauge treatment response and detect complications like hyperparathyroidism or nephrocalcinosis[^3][^7]. Radiographic monitoring (e.g., annual limb X-rays in children) tracks bone healing or deformities[^3]. Growth velocity charts in pediatric patients and dual-energy X-ray absorptiometry (DXA) scans in adults may supplement monitoring[^7]. Documentation often notes dose adjustments based on ALP trends or symptom progression[^3].""",
        "P": """**P — Plan**: Care plans commonly outline **medication optimization** (e.g., "increase phosphate to 60 mg/kg/day if ALP remains elevated"), **dietary modifications** (phosphate-rich foods), and **surgical referrals** for severe skeletal deformities[^1][^7]. Prophylactic dental care and **genetic counseling** for family members are frequently included[^1][^7]. Guidelines recommend documenting intent to transition pediatric patients to adult dosing regimens or consider burosumab for refractory cases[^3][^7]. Plans may also specify monitoring intervals (e.g., "repeat renal ultrasound in 6 months to assess for nephrocalcinosis")[^3].""",
        "E": """**E — Evaluation**: Progress is evaluated via **normalization of ALP**, **improved phosphate retention** (TmP/GFR), and **radiographic evidence of bone healing**[^3][^7]. Documentation may note "reduced leg bowing on X-ray" or "decreased bone pain with current regimen"[^1][^7]. In adults, stabilization of pseudofractures or improved mobility metrics (e.g., 6-minute walk test) are tracked[^7]. Persistent hypophosphatemia or complications (e.g., hyperparathyroidism) trigger reassessment of therapy[^3]. Pediatric growth curves and dentition assessments provide additional outcome measures[^1][^7].""",
        "R": """**R — Referral** : Common referrals include **nephrology** (renal complications), **endocrinology** (refractory hypophosphatemia), **orthopedics** (deformity correction), and **dentistry** (abscess prevention)[^1][^7]. Genetic counseling referrals are standard for family planning or testing asymptomatic relatives[^1][^7]. Rarely, patients with atypical presentations may be referred to metabolic bone centers for advanced diagnostics (e.g., FGF23 assays or genetic panels)[^3][^7]. Documentation typically justifies referrals (e.g., "orthopedic evaluation for worsening genu varum")[^1]."""
    }
icd_code = "E83.31"
# is_valid_suspect = result.get_is_valid_suspect(icd=icd_code, icd_description=icd_description)
# print("Is valid suspect:", is_valid_suspect)


