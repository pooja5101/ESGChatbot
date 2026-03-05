import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 1. Initialize Google Embeddings
# 'embedding-001' is highly efficient for ESG text retrieval
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# 2. Define the Vector Store (The Child Chunk Store)
# We point this to the 'chroma_db' folder in your project root
# This stores the vectors of the small 400-character chunks
persist_directory = os.path.join(os.path.dirname(__file__), "../../chroma_db")

vectorstore = Chroma(
    collection_name="esg_hierarchical_data",
    embedding_function=embeddings,
    persist_directory=persist_directory
)

# 3. Define the Document Store (The Parent Chunk Store)
# This stores the full 2000-character parent text. 
# For local dev, we use InMemoryStore. 
# For production/scale, replace this with LocalFileStore or a Redis store.
docstore = InMemoryStore()

# 4. Define the Hierarchical Splitters
# Parent: Provides broad context (e.g., an entire section on Carbon Targets)
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000, 
    chunk_overlap=200
)

# Child: Provides high-precision search matches
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400, 
    chunk_overlap=50
)

# 5. Build the Hierarchical Retriever
# This object manages the relationship: 
# It searches 'vectorstore' for children, then grabs the parent from 'docstore'.
hierarchical_retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=docstore,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# Example of how to use metadata filtering with this retriever:

# hierarchical_retriever.invoke("carbon emissions", filter={"company": "Apple"})

# In src/retrieval/retriever.py
#from langchain.storage import LocalFileStore

# Create a folder for the parent chunks
#parent_store_path = os.path.join(os.path.dirname(__file__), "../../parent_store")
#os.makedirs(parent_store_path, exist_ok=True)

# Use LocalFileStore so parent docs stay on your hard drive
#docstore = LocalFileStore(parent_store_path)