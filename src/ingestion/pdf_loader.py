import json
import hashlib
from pathlib import Path

import fitz

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")

def extract_pdf(pdf_path: Path) -> list[dict]:
    """Extract text while preserving page-level metadata"""

    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported!")

    document_id = hashlib.sha256(
        pdf_path.read_bytes()
    ).hexdigest()[:16]

    pages = []

    with fitz.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=2):
            text = page.get_text("text").strip()

            if not text:
                continue

            pages.append({
                "document_id": document_id,
                "document_name": pdf_path.name,
                "page_number": page_number,
                "text": text
            })

    return pages


def save_pages(pages: list[dict], output_path: Path):
    "Save extracted pages as JSON lines"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        for page in pages:
            file.write(json.dumps(page, ensure_ascii=False))
            file.write("\n")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = list(RAW_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in data/raw/")
        return


    for pdf_path in pdf_files:
        pages = extract_pdf(pdf_path)

        output_path = OUTPUT_DIR/f"{pdf_path.stem}.jsonl"

        save_pages(pages, output_path)

        print(
            f"{pdf_path.name}: "
            f"{len(pages)} pages extracted and saved to {output_path}"
        )

if __name__ == "__main__":
    main()  

