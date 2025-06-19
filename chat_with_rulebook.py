# AI Support Chatbot using Local Embeddings (CPU-only) with OpenAI LLMs — Enhanced Intelligence + Error Handling + Chroma Vector Store + Persistent Loading + Logging Conversations + Sensitive Data Masking + Reindexing Control

import os
import re
import logging
from dotenv import load_dotenv
from datetime import datetime
from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredWordDocumentLoader, JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("system_log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Config
PERSIST_DIR = "chroma_db"
LOG_FILE = "chat_log.txt"

# Function to mask sensitive information
def mask_sensitive(text):
    text = re.sub(r'(?i)(password\s*[:=]\s*)([^\s]+)', r'\1******', text)
    text = re.sub(r'(?i)(token\s*[:=]\s*)([^\s]+)', r'\1******', text)
    text = re.sub(r'(?i)(api[_-]?key\s*[:=]\s*)([^\s]+)', r'\1******', text)
    return text

# Step 1: Load all documents (PDF, Word, JSON)
def load_all_documents(pdf_paths, docx_paths, json_paths):
    docs = []
    try:
        for path in pdf_paths:
            docs.extend(PyMuPDFLoader(path).load())
        for path in docx_paths:
            docs.extend(UnstructuredWordDocumentLoader(path).load())
        for path in json_paths:
            docs.extend(JSONLoader(path, jq_schema=".").load())
        logger.info("Successfully loaded all documents.")
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        raise
    return docs

# Replace with your actual files
pdf_files = ["data_source/books/The_Rámáyan_of_Válmíki.pdf"]
docx_files = []
json_files = []

try:
    # Step 2: Load or create vectorstore
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if os.path.exists(PERSIST_DIR):
        try:
            logger.info("Loading existing Chroma vectorstore from disk...")
            vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding_model)
            logger.info(f"Chroma vectorstore contents: {vectorstore._collection.count()} documents")
        except Exception as e:
            logger.error(f"Failed to load Chroma DB: {e}")
            raise RuntimeError("Chroma DB might be corrupted. Please rebuild it or check file integrity.")
    else:
        logger.info("Creating new Chroma vectorstore from documents...")
        documents = load_all_documents(pdf_files, docx_files, json_files)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(chunks, embedding_model, persist_directory=PERSIST_DIR)

        logger.info("Chroma vectorstore created and persisted.")

    # Step 4: Build Conversational QA Chain with Memory using OpenAI LLM
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True,
        output_key="answer"
    )

    # Step 5: CLI Chat Interface with File Logging
    print("\n AI Support Bot Ready! Type 'exit' to quit.\n")
    chat_history = []
    with open(LOG_FILE, "a") as log:
        log.write(f"\n\n=== New Chat Session: {datetime.now()} ===\n")
        while True:
            try:
                query = input("You: ")
                if query.lower() in ["exit", "quit"]:
                    break
                result = qa_chain.invoke({"question": query, "chat_history": chat_history})
                answer = result["answer"]
                print("Bot:", answer)
                chat_history.append((query, answer))

                # Log masked version
                log.write(f"You: {mask_sensitive(query)}\n")
                log.write(f"Bot: {mask_sensitive(answer)}\n")
            except Exception as e:
                logger.error(f"Error during chat interaction: {e}")
                print("Bot: Sorry, something went wrong while processing your request.")
                log.write(f"Error: {e}\n")
except Exception as e:
    logger.critical(f"Critical failure: {e}")
    print("Startup failed. Check logs for more information.")
