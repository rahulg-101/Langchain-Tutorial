from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langserve import add_routes
import uvicorn
import os
from langchain_community.llms import Ollama
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Set the OPENAI_API_KEY environment variable from the loaded .env file
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

# Initialize the FastAPI app with metadata
app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple API Server"
)

# Add a route for the OpenAI chat model to the FastAPI app
add_routes(
    app,
    ChatOpenAI(),
    path="/openai"
)

# Initialize the OpenAI chat model
model = ChatOpenAI()

# Initialize the Ollama model with the "gemma:2b" model
llm = Ollama(model="gemma:2b")

# Create prompt templates for generating essays and poems
"""The `ChatPromptTemplate.from_template()` function is used in the code to create instances of 
ChatPromptTemplate based on a template string. These instances are designed to generate
dynamic chat prompts that can be used to interact with language models.
The purpose of using this function is twofold:

1. Dynamic Prompt Generation: It allows for the creation of prompts that can be customized with specific 
details at runtime. For example, in the provided code, {topic} in the template strings "Write me an essay 
about {topic} with 100 words" and "Write me a poem about {topic} for a 5 years child with 100 words" can 
be replaced with actual topics dynamically. This enables the application to request the language model to 
generate content based on varying topics without hardcoding the prompts for each possible topic.

2. Ease of Use and Flexibility: By defining prompts through templates, developers can easily modify the prompt 
structure without altering the underlying logic that handles the interaction with the language models. 
This makes the code more maintainable and flexible, as changes to the prompt requirements do not 
require significant code changes."""

prompt1 = ChatPromptTemplate.from_template("Write me an essay about {topic} with 100 words")
prompt2 = ChatPromptTemplate.from_template("Write me a poem about {topic} for a 5 years child with 100 words")

# Add a route for generating essays using the OpenAI chat model
add_routes(
    app,
    prompt1 | model,
    path="/essay"
)

# Add a route for generating poems using the Ollama model
add_routes(
    app,
    prompt2 | llm,
    path="/poem"
)

# Run the FastAPI app when the script is executed directly
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)