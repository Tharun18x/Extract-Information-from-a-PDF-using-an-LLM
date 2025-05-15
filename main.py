import pdfplumber
import numpy as np
import faiss
import openai
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text() + '\n'
    return text

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_text(text)

def create_faiss_index(text_chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(text_chunks)
    
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    
    return model, index, embeddings

def retrieve_relevant_chunks(query, model, index, chunks, top_k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    return [chunks[i] for i in indices[0]]

def generate_answer(context_chunks, question):
    context = "\n\n".join(context_chunks)
    prompt = f"Answer the question using the context below:\n\n{context}\n\nQuestion: {question}\nAnswer:"
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content'].strip()


def chat_with_pdf(pdf_path, question):
    print(" Extracting text...")
    text = extract_text_from_pdf(pdf_path)
    
    print(" Splitting into chunks...")
    chunks = split_text(text)
    
    print(" Creating FAISS index...")
    model, index, _ = create_faiss_index(chunks)
    
    print("Retrieving context...")
    relevant_chunks = retrieve_relevant_chunks(question, model, index, chunks)
    
    print(" Generating answer...")
    answer = generate_answer(relevant_chunks, question)
    
    return answer


pdf_path = r"" #path

question = "What are the main conclusions of this document?"

answer = chat_with_pdf(pdf_path, question)
print("\n Answer:", answer)

# from openai import OpenAI

# client = OpenAI(api_key="X")  # replace with your key

# response = client.chat.completions.create(
#     model="gpt-3.5-turbo",
#     messages=[
#         {"role": "user", "content": "What is Retrieval-Augmented Generation?"}
#     ]
# )

# print(response.choices[0].message.content)
