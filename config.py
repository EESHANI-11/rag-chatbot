import os

PDF_LOCAL_PATH = os.getenv("PDF_LOCAL_PATH", "data/ingenia_r11.1.pdf")
PDF_NAME = os.getenv("PDF_NAME", "Ingenia R11.1 - Technical Description")
PDF_URL = os.getenv("PDF_URL", "https://www.documents.philips.com/assets/Technical%20Description/20250729/11c095ff3f37409fb9e0b32900543b1b.pdf?feed=ifu_docs_feed")  # optional

CHROMA_DIR = os.getenv("CHROMA_DIR", ".chroma")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "ingenia_r11_techdesc")

# Groq
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
