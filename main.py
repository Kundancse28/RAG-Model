import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from firebase_admin import credentials, initialize_app, firestore
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain.docstore.document import Document


# Load environment variables
load_dotenv()

# Initialize Firebase
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "ragmodel-a7010-firebase-adminsdk-ho7zn-06ef494952.json"
cred = credentials.Certificate("ragmodel-a7010-firebase-adminsdk-ho7zn-06ef494952.json")
#initialize_app(cred)

pc = Pinecone(
    api_key="1eec320b-eeb9-4c79-8a84-a95e1b63c378"
)
index_name = 'langchainvector'

# Check if the index exists, and create it if necessary
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name='langchainvector',
        dimension=768,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )


# Helper function to read and split PDF text
def get_pdf_text(pdf):
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)


# Helper function to index document into Pinecone
def create_pinecone_index(chat_name, text_chunks):
    index = pc.Index(index_name)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Embed each document chunk (which gives a list of vectors)
    vectors = embeddings.embed_documents(text_chunks)
    
    upsert_data = []
    for i, vector in enumerate(vectors):
        upsert_data.append({
            "id": f"{chat_name}_{i}",  # Unique ID for each chunk
            "values": vector,  # Embedding values (vector)
            "metadata": {"text": text_chunks[i]}  # Metadata, including the chunked text
        })
    
    # Upsert the vectors into Pinecone
    index.upsert(vectors=upsert_data)


    
    # Store index metadata in Firebase
    db = firestore.client()
    db.collection("document_indices").document(chat_name).set({
        "index_name": index_name
    })

# Helper function to validate questions
def validate_question(question):
    if len(question) < 5:
        return False, "Question too short"
    if any(offensive_word in question.lower() for offensive_word in ["badword1", "badword2"]):
        return False, "Offensive content detected"
    return True, "Valid question"


# Helper function to get the conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the provided context, 
    say 'answer is not available in the context'.\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


# Streamlit App UI
st.title("Document Chat System")

# Upload PDF
st.header("Upload Document")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
chat_name = st.text_input("Chat Name")

if uploaded_file and chat_name:
    if st.button("Upload and Index Document"):
        try:
            # Extract text from the PDF and split into chunks
            text = get_pdf_text(uploaded_file)
            text_chunks = get_text_chunks(text)

            # Index the text chunks in Pinecone
            create_pinecone_index(chat_name, text_chunks)
            st.success("Document indexed successfully")
        except Exception as e:
            st.error(f"Error during document upload: {str(e)}")

# Ask Question
st.header("Ask a Question")
query_chat_name = st.text_input("Chat Name for Query")
question = st.text_input("Your Question")

if query_chat_name and question:
    if st.button("Submit Query"):
        try:
            # Validate the question first
            is_valid, message = validate_question(question)
            if not is_valid:
                st.error(message)
            else:
                # Retrieve the index from Firebase and query Pinecone
                db = firestore.client()
                index_ref = db.collection("document_indices").document(query_chat_name).get()
                if not index_ref.exists:
                    st.error("Document index not found")
                else:
                    index_name = index_ref.to_dict().get("index_name")
                    index = pc.Index(index_name)

                    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                    vector_query = embeddings.embed_query(question)
                    search_result = index.query(
                            vector=vector_query,  # Query vector
                            top_k=3,  # Number of top results to return
                            include_values=False,  # We don't need the vector values, just metadata
                            include_metadata=True  # We want to retrieve the associated metadata
                        )

                    docs = [Document(page_content=result['metadata']['text']) for result in search_result['matches'] if 'metadata' in result]
                    chain = get_conversational_chain()
                    response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)

                    st.success(f"Answer: {response['output_text']}")
        except Exception as e:
            st.error(f"Error: {str(e)}")