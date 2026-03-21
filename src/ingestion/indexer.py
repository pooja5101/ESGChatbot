import os
import uuid
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter
from src.retrieval.retriever import hierarchical_retriever
from dotenv import load_dotenv

load_dotenv()

import shutil

def reset_database(persist_directory="D:/Projects/AiProjects/ESGChatbot/chroma_db", store_directory="D:/Projects/AiProjects/ESGChatbot/data/parent_store"):
    """
    Completely wipes the Vector Database and the Parent Document Store.
    Ensures no 'ghost' documents or mismatched IDs remain.
    """
    print("--- 🧹 STARTING DATABASE RESET ---")
    
    # 1. Clear LocalFileStore (Parent Text)
    if os.path.exists(store_directory):
        shutil.rmtree(store_directory)
        print(f"Successfully deleted Parent Store at: {store_directory}")
    else:
        print("Parent Store folder not found, skipping...")

    # 2. Clear ChromaDB (Child Vectors)
    if os.path.exists(persist_directory):
        shutil.rmtree(persist_directory)
        print(f"Successfully deleted ChromaDB at: {persist_directory}")
    else:
        print("ChromaDB folder not found, skipping...")

    print("--- ✅ RESET COMPLETE: Ready for clean indexing ---")

# --- USAGE ---
# reset_database()

def batch_iterate(size, iterable):
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]

def ingest_esg_report(markdown_path: str, company_name: str, year: int, sector: str):
    """
    Reads a Markdown ESG report and indexes it into the hierarchical retriever.
    """
    #reset_database()
    if not os.path.exists(markdown_path):
        print(f"Error: File {markdown_path} not found.")
        return

    with open(markdown_path, "r", encoding="utf-8") as f:
        markdown_content = f.read()

    print(f"--- 🧱 Processing {company_name} ({year}) ---")

    # 1. Split Markdown by Headers (# and ##)
    # This ensures that a single 'Parent' chunk doesn't mix two different topics
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
    ]
    
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    sections = md_splitter.split_text(markdown_content)

    # 2. Enrich with Metadata
    # This metadata is what our 'Router Node' uses to filter the search
    final_docs = []
    for section in sections:
        metadata = {
            "company": company_name,
            "year": str(year),
            "sector": sector,
            "source": os.path.basename(markdown_path)
        }
        # Merge existing header metadata with our custom tags
        section.metadata.update(metadata)
        final_docs.append(section)

    # 3. Add to Hierarchical Retriever
    # This automatically triggers the parent/child splitting defined in retriever.py
    print(f"Indexing {len(final_docs)} logical sections into ChromaDB...")

    for batch in batch_iterate(100, final_docs): # Small batches are safer
        try:
        # CRITICAL: Call the retriever directly. 
        # Do NOT pass 'ids=' here; let the retriever generate them 
        # so it can map multiple children to one parent correctly.
            hierarchical_retriever.add_documents(batch)
        except Exception as e:
            print(f"Error indexing batch: {e}")
    
    print(f"✅ Successfully indexed {company_name} ESG data.")

# Example Standalone Execution:
if __name__ == "__main__":
    # In a real workflow, this would be called after your Gemini Vision parsing step
    ingest_esg_report("data/parsed_markdown/Mahindra-and-Mahindra-Sustainability-Report-2025.md", "Mahindra and Mahindra", 2025, "Automotive")
    #pass