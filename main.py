import pandas as pd
import numpy as np
import os
import pypdf
import docx2txt
import yaml
from langchain_community.document_loaders import PyPDFLoader
import requests





def yaml_parser(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def read_pdf_or_docx_file(file_path):
    if file_path.endswith('.pdf'):
        loader = pypdf.PdfReader(file_path)
        return loader.pages, "pdf"
    elif file_path.endswith('.docx'):
        return docx2txt.process(file_path), "docx"
    else:
        raise ValueError("Unsupported file format")
    
def divide_cv_sections(cv_text,file_format):
    """divide the text into sections beginning with uppercase letters"""
    sections = []
    current_section = []
    if file_format=='docx':
        for line in cv_text.split('\n'):
            if line and line[0].isupper():
                if current_section:
                    sections.append('\n'.join(current_section))
                current_section = [line]
            else:
                current_section.append(line)
        if current_section:
            sections.append('\n'.join(current_section))
    elif file_format=='pdf':
        for page in cv_text:
            for line in page.extract_text().split('\n'):
                if line.isupper():
                    if current_section:
                        sections.append('\n'.join(current_section))
                    current_section = [line]
                else:
                    current_section.append(line)
        if current_section:
            sections.append('\n'.join(current_section))
    return sections

def convert_sections_into_documents(cv_sections):
    """Divide the CV sections into separate langchain documents along with metadata as the section header."""
    documents = []
    for i, section in enumerate(cv_sections):
        doc = {
            "id": i,
            "content": section.split('\n')[1:],  # Exclude the section title from content
            "metadata": {
                "section_title": section.split('\n')[0]
            }
        }
        documents.append(doc)
    return documents

def embed_and_store_documents(documents):
    """Embed the documents and store them in a vector database."""
    for doc in documents:
        # Embed the document content
        embedding = embed_text('\n'.join(doc['content']))
        # Store the embedding along with metadata
        store_embedding(embedding, doc['metadata'])



yaml_file = yaml_parser("config.yaml")

cv_path = yaml_file['input']['cv_path']
model_id = yaml_file['params']['embedding_model']
hf_token = yaml_file['params']['hugging_face_token']

data,file_format = read_pdf_or_docx_file(cv_path)

extracted_data = divide_cv_sections(data,file_format)

cv_documents = convert_sections_into_documents(extracted_data)

api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
headers = {"Authorization": f"Bearer {hf_token}"}





# loader = PyPDFLoader(cv_path)

# print(data)
# cv= loader.load()

