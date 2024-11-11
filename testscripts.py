from PyPDF2 import PdfReader
from docx import Document
import os
import json
import io
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def read_pdf(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""  # Avoid NoneType errors
    return text

def read_txt(txt_docs):
    text = ""
    for txt in txt_docs:
        text += txt.read().decode("utf-8")  # Decode bytes to string
    return text

def read_docx(docx_docs):
    text = ""
    for docx in docx_docs:
        doc = Document(docx)
        for para in doc.paragraphs:
            text += para.text + "\n"  # Add line breaks
    return text

def read_pdf_from_directory(directory):
    text = ""
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(directory, filename)
            pdf_reader = PdfReader(pdf_path)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
    return text

def read_txt_from_directory(directory):
    text = ""
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            txt_path = os.path.join(directory, filename)
            with open(txt_path, 'r', encoding='utf-8') as file:
                text += file.read() + "\n"  # Add line breaks
    return text

def read_docx_from_directory(directory):
    text = ""
    for filename in os.listdir(directory):
        if filename.endswith('.docx'):
            docx_path = os.path.join(directory, filename)
            doc = Document(docx_path)
            for para in doc.paragraphs:
                text += para.text + "\n"  # Add line breaks
    return text

def initialize_generative_model():
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5)

def understand_tone_and_language(script_text):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts([script_text], embedding=embeddings)
    vector_store.save_local("faiss_index")


def generate_script(user_input, generative_model, reference_context):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    # Perform a similarity search with the user input
    script_docs = new_db.similarity_search(user_input)

    # Construct the prompt string
    prompt = f"""
    Take the persona of a Test Script Generator for different kinds of SAP modules.
    In the context, test scripts are given as a reference.
    based on the structure in the references generate a new test script using the context and test case
    Generate a test script strictly based on the test case and purpose/overview of the context 
    For Fiori Codes based on the test script generated understand it and return the Fiori Codes from the context
    Strictly dont use pre-existing test scripts as a response for a test case 
    Do not generate the same test script for two different test cases
    For Example the Structure of the Test Script 

         {{
    "Intracompany STO": {{
        "Step Description": {{
            "1": "Login to SAP Fiori...",
            "2": "Access the App...",
            ...
        }},
        "Expected Results": {{
            "1": "SAP Fiori Launchpad is displayed",
            "2": "Purchase Order for STO...",
            ...
        }},
        "Fiori ID":"F0842A"
    }}
    }}
    Output Should be strictly in the form of json
    
    Reference Context: \n{reference_context}\n
    User Input: \n{user_input}\n
    """

    # Load the QA chain with the correct chain type
    chain = load_qa_chain(generative_model, chain_type="refine")

    # Prepare the input for the chain, including the required 'question' key
    input_data = {
        "question": prompt,  # Pass the constructed prompt as the question
        "input_documents": script_docs,  # Use the similar documents found
    }

    # Invoke the chain using the new method
    response = chain.invoke(input_data)
    output_text = response.get("output_text", "").strip()
        # Remove code block markers (```json ... ```)
    if output_text.startswith("```json") and output_text.endswith("```"):
        output_text = output_text[7:-3].strip()
    response=json.loads(output_text)

    return response



def script(uploaded_text,case_description):
    references_directory = "References"
    combined_steps_file = os.path.join(references_directory, "combined_steps.json")
    fiori_codes_file = os.path.join(references_directory, "Fioricodes.json")

    # Load combined_steps.json
    with open(combined_steps_file, 'r') as file:
        reference_context = json.load(file)

    # Load Fioricodes.json and merge
    with open(fiori_codes_file, 'r') as file:
        fiori_codes_context = json.load(file)

    # Merge dictionaries (assuming both files contain JSON objects)
    reference_context.update(fiori_codes_context)
    

    reference_text = read_pdf_from_directory(references_directory)
    reference_text += read_txt_from_directory(references_directory)
    reference_text += read_docx_from_directory(references_directory)

    if reference_text:
        understand_tone_and_language(reference_text)
    query="Generate Test Cases by understanding the purpose of the doc uploaded"
    if uploaded_text and query:
        generative_model = initialize_generative_model()
        understand_tone_and_language(uploaded_text + reference_text)
        test_scripts={}
        test_scripts[case_description] = generate_script(case_description, generative_model, reference_context)
        return test_scripts[case_description]

        