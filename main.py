from langchain_community.document_loaders import PyPDFLoader
import pandas as pd
import numpy as np
import os
import pypdf
import docx2txt


print(np.arange(10))



# text_file_path=r"C:\Users\Samarth\OneDrive\Documents\CV\Samarth_Dhawan_CV.docx"
# # my_text = docx2txt.process(text_file_path)
# # print(my_text)

# data = resumeparse.read_file(text_file_path)

# print(data)
# # creating a pdf reader object

# cv_path="C:\\Users\\Samarth\\OneDrive\\Documents\\CV\\Data Scientist\\Samarth_Dhawan_CV.pdf"

# print("Loading CV from:", cv_path)

# reader = pypdf.PdfReader(cv_path)

# # print the number of pages in pdf file
# print(len(reader.pages))

# # print the text of the first page
# print(reader.pages[0].extract_text())



# loader = PyPDFLoader(cv_path,headers={'header 1' : 'experience'})

# docs = loader.load()
# print(docs[0])
# print(len(docs))