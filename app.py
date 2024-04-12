# Loading required libraries
import os
from langchain.chains import RetrievalQA
from langchain.memory import ConversationSummaryMemory
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import TextLoader

import ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

#select model
llm_model_name = "codellama"

#load data
data_path = "./data/Alpaca_Documentation.pdf"
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=30,
    length_function=len,)

documents = PyPDFLoader(data_path).load_and_split(text_splitter=text_splitter)

# Create Ollama embeddings and vector store
embeddings = OllamaEmbeddings(model=llm_model_name)
vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings)


# Create the retriever
retriever = vectorstore.as_retriever()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Define the Ollama LLM function
def ollama_llm(question, context):
    formatted_prompt = f"Question: {question}\n\nContext: {context}"
    response = ollama.chat(model=llm_model_name, messages=[{'role': 'user', 'content': formatted_prompt}])
    return response['message']['content']

# Define the RAG chain
def rag_chain(question):
    retrieved_docs = retriever.invoke(question)
    formatted_context = format_docs(retrieved_docs)
    return ollama_llm(question, formatted_context)

if __name__ in "__main__":
    # Use the RAG chain
    result = rag_chain("how old is the united states")
    print(result)
    pass










