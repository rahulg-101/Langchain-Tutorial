"""
When this code is run, it creates a web application using Streamlit that allows users to input a question. 
The application then uses the Langchain and LLAMA2 API to process this question, generating a response based on the Gemma:2B model. 
The response is parsed into a string and displayed on the web application. The environment variables for Langchain API keys are set up 
to authenticate API requests, and Langchain tracing is enabled for tracking purposes. The initial import of ChatOpenAI is unnecessary 
for this script's functionality and could be removed to clean up the code.
"""    


# Importing the ChatOpenAI class is not used in this snippet and can be removed.
from langchain_openai import ChatOpenAI  

# Importing ChatPromptTemplate to create structured prompts for the chat model.
from langchain_core.prompts import ChatPromptTemplate  

# Importing StrOutputParser to parse the output from the chat model into a string.
from langchain_core.output_parsers import StrOutputParser  

# Importing Ollama for interacting with the LLAMA2 model.
from langchain_community.llms import Ollama  

# Importing streamlit to create a web app interface.
import streamlit as st  

# Importing the os module for operating system interactions, like managing environment variables.
import os  

# Importing load_dotenv to load environment variables from a .env file.
from dotenv import load_dotenv  

load_dotenv()  # Loading environment variables from a .env file into the os.environ dictionary.

os.environ["LANGCHAIN_TRACING_V2"]="true"  # Enabling Langchain tracing by setting the LANGCHAIN_TRACING_V2 environment variable to true.
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")  # Setting the LANGCHAIN_API_KEY environment variable by retrieving it from the environment.

## Prompt Template
prompt=ChatPromptTemplate.from_messages(  # Creating a prompt template with predefined messages for the chat model.
    [
        ("system","You are a helpful assistant. Please response to the user queries"),  # System message to define the role of the assistant.
        ("user","Question:{question}")  # User message template to include the user's question in the prompt.
    ]
)

## streamlit framework

st.title('Langchain Demo With Gemma:2B API')  # Setting the title of the Streamlit web app.
input_text=st.text_input("Search the topic u want")  # Creating a text input field in the Streamlit app for users to enter their query.

# ollama LLAma2 LLm 
llm=Ollama(model="gemma:2b")  # Initializing the Ollama class with the LLAMA2 model for processing the input text.
output_parser=StrOutputParser()  # Initializing the StrOutputParser to convert the chat model's output into a string.
chain=prompt|llm|output_parser  # Creating a processing chain that combines the prompt, the chat model (LLAMA2), and the output parser.

if input_text:  # Checking if the user has entered any text in the input field.
    st.write(chain.invoke({"question":input_text}))  # Invoking the processing chain with the user's question and displaying the result in the Streamlit app.    


