# allymar/MemberRag.py

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date
import os
from typing import List, Any, Dict, Optional, Union
from retry import retry
import vertexai
from helper import df_from_bigquery, get_file, split_documents, combine_queries_with_kg
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
from langchain_google_community import BigQueryVectorStore
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from dotenv import load_dotenv
from pdf2image import pdfinfo_from_path, convert_from_path
import gc
import tempfile
from google.genai import types
from google import genai

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
client = genai.Client(vertexai=True, project=PROJECT_ID, location=REGION)

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
patient_name = "Amelia Harris"


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
                    "source": row["source"],
                    "chunk": row["chunk"],
                    "file_path": row["file_path"],
                    "patient_name": row["patient_name"],
                }
            )
            for _, row in df.iterrows()
        ] 
        return documents

    def ocr(self, pdf_path, page_number, dpi=300):
        print("Performing OCR on page number:", page_number)
        images = convert_from_path(pdf_path, dpi=dpi, first_page=page_number, last_page=page_number)
        image = images[0]

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_image_file:
            image.save(temp_image_file.name, format="PNG")
            temp_image_file_path = temp_image_file.name

        try:

            with open(temp_image_file_path, 'rb') as f:
                image_bytes = f.read()

            response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=[
                types.Part.from_bytes(
                data=image_bytes,
                mime_type='image/jpeg',
                ),
                'Extract the text in the image verbatim'
            ],
            )
            response=f"\n PAGE NUMBER:- {page_number}----------------------------------------\nDATA: {response.text}"
            return response
        finally:
            # Clean up
            os.remove(temp_image_file_path)
            del image
            gc.collect()

    # @retry(max_retries=5, backoff_factor=.5, verbose=True)
    def add_documents_to_index(self, file_name: str, metadata: dict= {}) -> bool:
        """Add documents to the index."""
        try:
            document_main=[]

            bq_file_name=f"LLM-Test/{self.member_id}/{file_name}"
            gcs_file_path = f"gs://{self.bucket}/{bq_file_name}"
            pdf_path = get_file(bucket_name=bucket, file_path=bq_file_name, local_file_name=file_name.split('/')[-1])
            info = pdfinfo_from_path(pdf_path)
            total_pages = info["Pages"]
            # total_pages = 10 
            for i in range(1, total_pages + 1):
                page_wise_extracted_data = self.ocr(pdf_path=pdf_path, page_number=i)
                document_main.append(page_wise_extracted_data)

            print("Deleting the temp file used in OCR...")
            # remove the downloaded file
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
            else:
                print(f"The file {pdf_path} does not exist")

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
            raise RuntimeError(f"Failed to verify/create index for member_id={self.member_id}: {e}")


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
            return rag_chain.invoke(question)
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
# SAMPLE DATA FOR THE KNOWLEDGE GRAPH
icd_code = "E83.31"
icd_description =  {
        "T": """**T — Treatment** :Documentation for familial hypophosphatemia (FH) typically reflects treatment with **oral phosphate salts** (e.g., 20–80 mg/kg/day elemental phosphorus split into 4–6 daily doses) and **active vitamin D analogs** like calcitriol (0.25–0.75 µg/day) to counteract renal phosphate wasting and impaired mineralization[^1][^3]. Since 2018, **burosumab** (Crysvita), a monoclonal antibody targeting FGF23, is administered subcutaneously every 2–4 weeks for X-linked hypophosphatemia (XLH)[^1][^7]. Documentation often specifies medication dosages, frequency, and adjustments based on lab monitoring or adverse effects (e.g., hypercalciuria, nephrocalcinosis)[^3]. Supportive measures, such as dental sealants to prevent abscesses, may also be noted[^1]. Guideline-concordant care emphasizes individualized regimens to balance biochemical correction with avoidance of complications[^3][^7].""",
        "A": """**A — Assessment**: Clinical documentation typically includes **hypophosphatemia** (serum phosphate below age-adjusted norms), **elevated FGF23**, and **normal calcium/25-hydroxyvitamin D levels**, alongside **renal phosphate wasting** (elevated urinary phosphate or reduced TmP/GFR)[^1][^3][^7]. Radiographs may show **rickets** (children) or **osteomalacia** (adults), such as metaphyseal fraying or pseudofractures[^3][^7]. Genetic testing confirming *PHEX* (XLH) or *FGF23* (ADHR) variants is increasingly documented to differentiate FH from acquired causes[^1][^7]. Notes often exclude nutritional deficiencies or secondary causes (e.g., tumor-induced osteomalacia) and may reference family history of skeletal abnormalities[^3][^7].""",
        "M": """**M — Monitoring** :Longitudinal documentation includes **serial serum phosphate**, **alkaline phosphatase (ALP)**, **urinary calcium/creatinine ratios**, and **renal function tests** every 3–6 months to gauge treatment response and detect complications like hyperparathyroidism or nephrocalcinosis[^3][^7]. Radiographic monitoring (e.g., annual limb X-rays in children) tracks bone healing or deformities[^3]. Growth velocity charts in pediatric patients and dual-energy X-ray absorptiometry (DXA) scans in adults may supplement monitoring[^7]. Documentation often notes dose adjustments based on ALP trends or symptom progression[^3].""",
        "P": """**P — Plan**: Care plans commonly outline **medication optimization** (e.g., "increase phosphate to 60 mg/kg/day if ALP remains elevated"), **dietary modifications** (phosphate-rich foods), and **surgical referrals** for severe skeletal deformities[^1][^7]. Prophylactic dental care and **genetic counseling** for family members are frequently included[^1][^7]. Guidelines recommend documenting intent to transition pediatric patients to adult dosing regimens or consider burosumab for refractory cases[^3][^7]. Plans may also specify monitoring intervals (e.g., "repeat renal ultrasound in 6 months to assess for nephrocalcinosis")[^3].""",
        "E": """**E — Evaluation**: Progress is evaluated via **normalization of ALP**, **improved phosphate retention** (TmP/GFR), and **radiographic evidence of bone healing**[^3][^7]. Documentation may note "reduced leg bowing on X-ray" or "decreased bone pain with current regimen"[^1][^7]. In adults, stabilization of pseudofractures or improved mobility metrics (e.g., 6-minute walk test) are tracked[^7]. Persistent hypophosphatemia or complications (e.g., hyperparathyroidism) trigger reassessment of therapy[^3]. Pediatric growth curves and dentition assessments provide additional outcome measures[^1][^7].""",
        "R": """**R — Referral** : Common referrals include **nephrology** (renal complications), **endocrinology** (refractory hypophosphatemia), **orthopedics** (deformity correction), and **dentistry** (abscess prevention)[^1][^7]. Genetic counseling referrals are standard for family planning or testing asymptomatic relatives[^1][^7]. Rarely, patients with atypical presentations may be referred to metabolic bone centers for advanced diagnostics (e.g., FGF23 assays or genetic panels)[^3][^7]. Documentation typically justifies referrals (e.g., "orthopedic evaluation for worsening genu varum")[^1]."""
    }

# Step 1: Initialize the RAG pipeline for the member
rag = MemberRAG()
file_name = "Amelia_Harris_Redacted_EDited.pdf"

# Step 3: Add the OCR-extracted content to the BigQuery index if not the pdf data is not indexed
success = rag.add_documents_to_index(file_name=file_name)

# Step 4: Ask a medical question
question = "What medications has the patient been prescribed for hypophosphatemia?"
answer = rag.query(question=question)

print("\nAnswer from RAG:")
print(answer)

suspect_decision = rag.get_is_valid_suspect(icd=icd_code, icd_description=icd_description)

print("\nWould you suspect the condition today?")
print(suspect_decision)

# Call the get_relevant_docs_and_metadata method
# data1 = rag.get_relevant_docs_and_metadata(question=question, k=DEFAULT_K, threshold=DEFAULT_THRESHOLD)


