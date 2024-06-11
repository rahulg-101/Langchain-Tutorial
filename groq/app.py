import streamlit as st
import os

# Import the ChatGroq model from langchain_groq.
# ChatGroq is a chat model provided by Groq, an AI hardware company.
from langchain_groq import ChatGroq

# Import WebBaseLoader from langchain_community.document_loaders.
# WebBaseLoader is used to load content from web pages.
from langchain_community.document_loaders import WebBaseLoader

# Import OllamaEmbeddings from langchain.embeddings.
# OllamaEmbeddings generates embeddings using the Ollama library.
from langchain_community.embeddings import OllamaEmbeddings


# Import RecursiveCharacterTextSplitter from langchain.text_splitter.
# This splitter divides text into chunks based on characters.
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import create_stuff_documents_chain from langchain.chains.combine_documents.
# This function creates a chain that combines multiple documents into one input for an LLM.
from langchain.chains.combine_documents import create_stuff_documents_chain

# Import ChatPromptTemplate from langchain_core.prompts.
# ChatPromptTemplate is used to create structured prompts for chat models.
from langchain_core.prompts import ChatPromptTemplate

# Import create_retrieval_chain from langchain.chains.
# This function creates a chain that retrieves documents and then processes them.
from langchain.chains import create_retrieval_chain

# Import FAISS from langchain_community.vectorstores.
# FAISS is a library for efficient similarity search and clustering of dense vectors.
from langchain_community.vectorstores import FAISS

# Import the time module for measuring execution time.
import time

# Import load_dotenv from dotenv to load environment variables from a .env file.
from dotenv import load_dotenv

# Load environment variables from a .env file in the current directory.
load_dotenv()
# os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

# Load the Groq API key from environment variables.
groq_api_key = os.environ['GROQ_API_KEY']

# Check if 'vector' is not in the Streamlit session state.
# This ensures that resource-intensive operations are done only once per session.
if "vector" not in st.session_state:
   # Initialize OllamaEmbeddings for generating embeddings.
   st.session_state.embeddings = OllamaEmbeddings(model = 'gemma:2b')
   
   # Create a WebBaseLoader to load content from LangSmith docs.
   st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/")
   
   # Load the web content into documents.
   st.session_state.docs = st.session_state.loader.load()

   # Create a RecursiveCharacterTextSplitter for chunking documents.
   st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
   
   # Split the first 50 documents into chunks.
   st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
   
   # Create a FAISS vector store from the document chunks using Ollama embeddings.
   st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Set the title of the Streamlit app.
st.title("ChatGroq Demo")

# Initialize the ChatGroq model with the API key and model name.
llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")

# Create a ChatPromptTemplate for generating context-aware responses.
prompt = ChatPromptTemplate.from_template(
   """
   Answer the questions based on the provided context only.
   Please provide the most accurate response based on the question
   <context>
   {context}
   <context>
   Questions:{input}
   """
)

# Create a chain that processes retrieved documents using the LLM and prompt.
document_chain = create_stuff_documents_chain(llm, prompt)

# Convert the FAISS vector store to a retriever for fetching relevant documents.
retriever = st.session_state.vectors.as_retriever()

# Create a chain that retrieves documents and then processes them with the document chain.
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Create a text input widget in Streamlit for user queries.
prompt = st.text_input("Input your prompt here")

# If a prompt is provided (i.e., not empty), process it.
if prompt:
   # Record the start time to measure response time.
   start = time.process_time()
   
   # Invoke the retrieval chain with the user's prompt.
   response = retrieval_chain.invoke({"input": prompt})
   
   # Calculate and print the response time.
   print("Response time :", time.process_time() - start)
   
   # Display the generated answer in the Streamlit app.
   st.write(response['answer'])

   # Create an expandable section in Streamlit to show document chunks.
   with st.expander("Document Similarity Search"):
       # Iterate through and display the content of each relevant document chunk.
       for i, doc in enumerate(response["context"]):
           st.write(doc.page_content)
           st.write("--------------------------------")