# Importing the ChatOpenAI class from langchain_openai module for interacting with OpenAI's API.
from langchain_openai import ChatOpenAI  

# Importing ChatPromptTemplate for creating structured prompts for the chat model.
from langchain_core.prompts import ChatPromptTemplate  

# Importing StrOutputParser to parse the output from the chat model into a string.
from langchain_core.output_parsers import StrOutputParser  

import streamlit as st  
import os  

# Importing load_dotenv from dotenv to load environment variables from a .env file.
from dotenv import load_dotenv  

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")  # Setting the OPENAI_API_KEY environment variable by retrieving it from the environment.

## Langmith tracking
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

st.title('Langchain Demo With OPENAI API')  # Setting the title of the Streamlit web app.
input_text=st.text_input("Search the topic u want")  # Creating a text input field in the Streamlit app for users to enter their query.

# openAI LLm 
llm=ChatOpenAI(model="gpt-3.5-turbo")  # Initializing the ChatOpenAI class with a specific model, in this case, GPT-3.5-turbo.
output_parser=StrOutputParser()  # Initializing the StrOutputParser to convert the chat model's output into a string.
chain=prompt|llm|output_parser  # Creating a processing chain that combines the prompt, the chat model, and the output parser.

if input_text:  # Checking if the user has entered any text in the input field.
    st.write(chain.invoke({'question':input_text}))  # Invoking the processing chain with the user's question and displaying the result in the Streamlit app.    


"""When this code is run, it creates a web application using Streamlit that allows users to input a question.
The application then uses the Langchain and OpenAI API to process this question, generating a response based on the GPT-3.5-turbo model.
The response is parsed into a string and displayed on the web application. The environment variables for OpenAI and 
Langchain API keys are set up to authenticate API requests, and Langchain tracing is enabled for tracking purposes."""    