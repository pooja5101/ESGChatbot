import os
from langchain.schema import Document
from langchain.text_splitter import MarkdownHeaderTextSplitter
from src.retrieval.retriever import hierarchical_retriever

def ingest_esg_report(markdown_path: str, company_name: str, year: int, sector: str):
    """
    Reads a Markdown ESG report and indexes it into the hierarchical retriever.
    """
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
    hierarchical_retriever.add_documents(final_docs)
    
    print(f"✅ Successfully indexed {company_name} ESG data.")

# Example Standalone Execution:
if __name__ == "__main__":
    # In a real workflow, this would be called after your Gemini Vision parsing step
    # ingest_esg_report("data/parsed_markdown/apple_2024.md", "Apple", 2024, "Technology")
    pass