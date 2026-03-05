import os
from pdf2image import convert_from_path
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import base64
from io import BytesIO

# Initialize Gemini 1.5 Pro for Vision tasks
# We use a low temperature (0) for factual extraction to prevent "hallucinating" numbers
vision_model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0)

def encode_image(image):
    """Helper to convert PIL image to base64 for the Gemini API."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def parse_pdf_to_markdown(pdf_path: str, output_folder: str):
    """
    Converts a PDF into a high-fidelity Markdown file using Gemini Vision.
    """
    if not os.path.exists(pdf_path):
        print(f"Error: {pdf_path} not found.")
        return

    # 1. Convert PDF pages to images
    # Note: Ensure Poppler is installed on your system (brew install poppler / apt-get install poppler-utils)
    print(f"--- 📄 Converting {os.path.basename(pdf_path)} to images ---")
    pages = convert_from_path(pdf_path, dpi=300) # 300 DPI is standard for high-quality OCR

    full_markdown_content = ""

    # 2. Process each page with Gemini Vision
    for i, page in enumerate(pages):
        print(f"Processing Page {i+1}/{len(pages)}...")
        
        base64_image = encode_image(page)
        
        # This prompt is optimized for ESG/Financial documents
        prompt = """
        Analyze this image of an ESG report page. 
        1. Extract all text while maintaining the hierarchical structure (use # for titles, ## for sections).
        2. VERY IMPORTANT: If there are tables, convert them into clean Markdown table format (| Column | Column |).
        3. Do not omit any financial or sustainability metrics (numbers, percentages, dates).
        4. If there is a chart, describe the key data points in bullet points.
        5. Output ONLY the markdown text. Do not include introductory comments or 'Here is the markdown'.
        """

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ]
        )

        response = vision_model.invoke([message])
        full_markdown_content += f"\n\n\n\n"
        full_markdown_content += response.content
        full_markdown_content += f"\n\n\n\n"

    # 3. Save the resulting Markdown
    file_name = os.path.basename(pdf_path).replace(".pdf", ".md")
    save_path = os.path.join(output_folder, file_name)
    
    os.makedirs(output_folder, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(full_markdown_content)
    
    print(f"✅ Success! Markdown saved to: {save_path}")
    return save_path

if __name__ == "__main__":
    # Example usage:
    # parse_pdf_to_markdown("data/raw_pdfs/apple_esg_2024.pdf", "data/parsed_markdown")
    pass