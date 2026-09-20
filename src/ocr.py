from io import BytesIO
MAX_FILE_SIZE = 10 * 1024 * 1024

def extract_document_text(uploaded_file):
    if uploaded_file is None: return {"text":"","method":"None","warning":"No document uploaded."}
    if getattr(uploaded_file,"size",0)>MAX_FILE_SIZE: raise ValueError("File exceeds 10 MB.")
    raw=uploaded_file.getvalue(); mime=getattr(uploaded_file,"type","")
    if mime=="application/pdf":
        try:
            from pypdf import PdfReader
            text="\n".join((p.extract_text() or "") for p in PdfReader(BytesIO(raw)).pages).strip()
            return {"text":text,"method":"PDF text extraction","warning":"" if text else "No embedded text found. Enter it manually below."}
        except Exception: return {"text":"","method":"PDF fallback","warning":"Could not extract PDF text. Enter it manually below."}
    try:
        from PIL import Image
        import pytesseract
        text=pytesseract.image_to_string(Image.open(BytesIO(raw)).convert("RGB")).strip()
        return {"text":text,"method":"Tesseract OCR","warning":"Review and correct the extracted text." if text else "No readable text detected. Enter it manually below."}
    except Exception:
        return {"text":"","method":"Manual OCR fallback","warning":"Automatic image OCR needs local Tesseract. Enter extracted text manually below."}
