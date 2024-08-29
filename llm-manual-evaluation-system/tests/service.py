from pypdf import PdfReader
import re


def load_PDF_text(path):
    loader = PdfReader(path)
    PDFtext = ""
    for page in loader.pages:
        PDFtext += " ".join(page.extract_text())
    return PDFtext


def reduce_extra_spaces(text):
    # Remove spaces between characters within words
    cleaned_text = re.sub(r"(?<=\w)\s(?=\w)", "", text)

    # Replace multiple spaces between words with a single space
    cleaned_text = re.sub(r"\s+", " ", cleaned_text)

    return cleaned_text
