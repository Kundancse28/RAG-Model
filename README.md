# Document Chat System

## Overview

The Document Chat System is a Streamlit web application that allows users to upload PDF documents, index their content using Pinecone, and interact with the documents using natural language queries. The system leverages Google Generative AI for embeddings and conversational responses, as well as Firebase for metadata storage.

The core functionalities include:
- Uploading and indexing PDF documents into Pinecone.
- Querying the indexed documents with natural language questions.
- Generating responses from the document content based on the queries.

## Features

- **Upload PDF Documents:** Extracts and splits the text from PDF files for indexing.
- **Indexing with Pinecone:** Stores document embeddings in a Pinecone vector database for efficient querying.
- **Natural Language Question Answering:** Users can submit questions related to the indexed document, and the system will provide detailed answers based on the document content.
- **Firebase Integration:** Stores metadata about indexed documents and supports retrieval of document indices for querying.

## Technologies Used

- **Streamlit:** Web framework for building the user interface.
- **PyPDF2:** For extracting text from PDF files.
- **Pinecone:** For vector indexing and retrieval of document embeddings.
- **Google Generative AI:** Used for both text embedding and conversational response generation.
- **Firebase (Firestore):** For storing and retrieving document metadata.
- **LangChain:** Provides conversational AI pipelines and question-answering chains.
- **dotenv:** For managing environment variables.

## Setup Instructions

### Prerequisites

Ensure you have the following installed:
- Python 3.8 or higher
- Pip
- Firebase Project with service account credentials
- Pinecone account and API key
- Google Cloud API for Generative AI

### Environment Variables

You need to create a `.env` file to store the following environment variables:

```bash
# .env file

# Pinecone API key
PINECONE_API_KEY=your-pinecone-api-key

# Google Cloud service credentials for Firebase
GOOGLE_APPLICATION_CREDENTIALS=path-to-your-firebase-admin-sdk.json
```

### Firebase Setup

- Download your Firebase project’s service account key and save it as `ragmodel-a7010-firebase-adminsdk-ho7zn-06ef494952.json`.
- Ensure Firestore is enabled in your Firebase project.

### Pinecone Setup

- Create a Pinecone account and retrieve your API key.
- Ensure you have set up the necessary vector index with the correct dimensions and metric (768 dimensions, cosine metric).

### Installation

1. Install the required Python packages:

```bash
pip install -r requirements.txt
```

2. Set up Firebase credentials and environment variables as explained above.

### Running the Application

After setting up the environment variables and Firebase credentials, run the application using Streamlit:

```bash
streamlit run app.py
```

### Uploading and Indexing Documents

1. Upload a PDF file through the web interface.
2. Provide a unique chat name for the document.
3. Click on "Upload and Index Document" to index the document in Pinecone.

### Querying the Document

1. Enter the chat name for the document you previously indexed.
2. Ask a natural language question related to the content of the document.
3. Receive a response generated from the document’s content.

## Folder Structure

```
|-- main.py                   # Main Streamlit application file
|-- requirements.txt          # Python package dependencies
|-- .env                      # Environment variables (hidden file)
|-- README.md                 # Project documentation
|-- ragmodel-a7010-firebase-adminsdk-ho7zn-06ef494952.json # Firebase credentials (Not included, must be added manually)
```

## Dependencies

The `requirements.txt` file contains all required dependencies. Install them with:

```bash
pip install -r requirements.txt
```

Example of `requirements.txt`:
```txt
        streamlit
        PyPDF2
        langchain
        firebase-admin
        python-dotenv
        pinecone-client
        langchain-google-genai
```

## Notes

- Ensure that you configure Google Generative AI access and install the necessary libraries for it to function.
- Be mindful of Pinecone usage limits based on your account plan.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Acknowledgments

- [LangChain](https://github.com/hwchase17/langchain)
- [Streamlit](https://github.com/streamlit/streamlit)
- [Pinecone](https://www.pinecone.io/)
- [Google Generative AI](https://cloud.google.com/ai/)