import os
import shutil
import PyPDF2
import csv
import chromadb
import uuid
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from ESGChatbot.src.chatbot_app.vector_store.agent.graph import build_esg_graph
from ESGChatbot.src.chatbot_app.vector_store.retrieval.retriever import hierarchical_retriever
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel



app = FastAPI()

# Load environment variables (like GOOGLE_API_KEY) from your .env file
load_dotenv() 

def main():
    agent = build_esg_graph()
    collection = hierarchical_retriever.vectorstore._collection
    count = collection.count()
    print(f"--- DEBUG: Total chunks in ChromaDB: {count} ---")
    keys = list(hierarchical_retriever.docstore.yield_keys())
    print(f"Total parent documents in store: {len(keys)}")
    print("🏦 Welcome to the ESG Banking Assistant. Type 'exit' to quit.")
    while True:
        user_input = input("\nQuery: ")
        if user_input.lower() == 'exit':
            break
            
        inputs = {"messages": [HumanMessage(content=user_input)]}
        
        # Run the agent
        final_state = agent.invoke(inputs)
        print(f"\nAnalyst:\n{final_state['messages'][-1].content}")

app = FastAPI()

class ChatRequest(BaseModel):
    message: str
    context_files: list[str] = []

@app.post("/api/chat")
async def chat(request: ChatRequest):
    # Mock ESG analysis response
    return {"response": "Based on the attached scope 3 emissions report...", "esg_score": 75}

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    # Mock document processing
    return {"filename": file.filename, "status": "processed", "insights": "Contains climate risk data."}


if __name__ == "__main__":
    main()



# ---------------------------------------------------------
# CHROMADB SETUP
# We use a persistent client so your data survives restarts.
# By default, it uses the 'all-MiniLM-L6-v2' embedding model.
# ---------------------------------------------------------
# chroma_client = chromadb.PersistentClient(path="./chroma_data")
# collection = chroma_client.get_or_create_collection(name="esg_documents")

# class ChatRequest(BaseModel):
#     message: str
#     context_files: list[str] = []

# # ---------------------------------------------------------
# # HELPER: TEXT EXTRACTION & CHUNKING
# # ---------------------------------------------------------
# def extract_text_from_file(file_path: str, filename: str) -> str:
#     # [Same implementation as before...]
#     text = ""
#     try:
#         if filename.lower().endswith(".pdf"):
#             with open(file_path, "rb") as f:
#                 reader = PyPDF2.PdfReader(f)
#                 for page in reader.pages:
#                     extracted = page.extract_text()
#                     if extracted:
#                         text += extracted + "\n"
#         elif filename.lower().endswith(".csv"):
#             with open(file_path, "r", encoding="utf-8") as f:
#                 reader = csv.reader(f)
#                 for row in reader:
#                     text += " | ".join(row) + "\n"
#         else:
#             with open(file_path, "r", encoding="utf-8") as f:
#                 text = f.read()
#     except Exception as e:
#         print(f"Error parsing {filename}: {e}")
#     return text

# def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
#     """Splits text into overlapping chunks for better vector retrieval."""
#     chunks = []
#     start = 0
#     while start < len(text):
#         end = start + chunk_size
#         chunks.append(text[start:end])
#         start += chunk_size - overlap # Move forward, leaving some overlap
#     return chunks

# # ---------------------------------------------------------
# # ENDPOINT: UPLOAD DOCUMENT
# # ---------------------------------------------------------
# @app.post("/api/upload")
# async def upload_document(file: UploadFile = File(...)):
#     if not file.filename:
#         raise HTTPException(status_code=400, detail="No file provided")

#     os.makedirs("temp_uploads", exist_ok=True)
#     file_path = f"temp_uploads/{file.filename}"

#     try:
#         # 1. Save locally (In production, save to AWS S3 here)
#         with open(file_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)

#         # 2. Extract and Chunk Text
#         extracted_text = extract_text_from_file(file_path, file.filename)
#         if not extracted_text.strip():
#             raise HTTPException(status_code=400, detail="Could not extract text.")
            
#         chunks = chunk_text(extracted_text)

#         # 3. Add to ChromaDB
#         # We need unique IDs, the chunk text, and metadata mapping it to the file
#         ids = [f"{file.filename}_{uuid.uuid4().hex[:8]}" for _ in chunks]
#         metadatas = [{"filename": file.filename} for _ in chunks]

#         collection.add(
#             documents=chunks,
#             metadatas=metadatas,
#             ids=ids
#         )

#         insights = "Contains specific carbon footprint data." if "scope 3" in extracted_text.lower() else "Document processed successfully."

#         return {
#             "filename": file.filename, 
#             "status": "processed", 
#             "insights": insights,
#             "chunks_stored": len(chunks)
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
#     finally:
#         file.file.close()
#         if os.path.exists(file_path):
#             os.remove(file_path)

# # ---------------------------------------------------------
# # ENDPOINT: CHAT (Retrieval Augmented Generation)
# # ---------------------------------------------------------
# @app.post("/api/chat")
# async def chat(request: ChatRequest):
#     context = ""
    
#     # 1. Query ChromaDB for relevant chunks if files are attached
#     if request.context_files:
#         # Ask ChromaDB to find the top 3 chunks most semantically similar to the user's question
#         # and ONLY look inside the specific files the user attached.
#         results = collection.query(
#             query_texts=[request.message],
#             n_results=3,
#             where={"filename": {"$in": request.context_files}}
#         )
        
#         # Assemble the retrieved chunks into our context string
#         if results['documents'] and results['documents'][0]:
#             retrieved_chunks = results['documents'][0]
#             for i, chunk in enumerate(retrieved_chunks):
#                 context += f"\n--- Excerpt {i+1} ---\n{chunk}\n"

#     # 2. Simulate the LLM Response
#     if context:
#         ai_reply = (
#             f"Based on a semantic search of your attached documents, I found relevant context. "
#             f"Here is what the vector database retrieved:\n\n{context[:300]}...\n\n"
#             f"[In production, an LLM would summarize this to answer: '{request.message}']"
#         )
#         esg_score = 82 # Mock score
#     else:
#         ai_reply = f"I am answering generally about '{request.message}' as no specific documents were found or attached."
#         esg_score = None

#     return {
#         "response": ai_reply, 
#         "esg_score": esg_score
#     }
