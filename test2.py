# python rag-test.py --ingest
# python rag-test.py

from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import sys
 
class ChatWebDoc:
    vector_store = None
    retriever = None
    chain = None
 
    def __init__(self):
        self.model = ChatOllama(model="mistral:instruct")
        #Loading embedding
        self.embedding = FastEmbedEmbeddings()
 
        self.text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=100)
        self.prompt = ChatPromptTemplate.from_messages(
        [
            ("system", 
"""You are an assistant for question-answering tasks. Use only the following 
context to answer the question. If you don't know the answer, just say that you don't know.
 
CONTEXT:
 
{context}
"""),
            ("human", "{input}"),
        ]
    )
 
    def ingest(self, url_list):
        #Load web pages
        docs = WebBaseLoader(url_list).load()
        chunks = self.text_splitter.split_documents(docs)
 
        #Create vector store
        vector_store = Chroma.from_documents(documents=chunks, 
            embedding=self.embedding, persist_directory="./chroma_db")
 
    def load(self):
        #Load vector store
        vector_store = Chroma(persist_directory="./chroma_db", 
            embedding_function=self.embedding)
 
        #Create chain
        self.retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.5,
            },
        )
 
        document_chain = create_stuff_documents_chain(self.model, self.prompt)
        self.chain = create_retrieval_chain(self.retriever, document_chain)
 
    def ask(self, query: str):
        if not self.chain:
            self.load()
 
        result = self.chain.invoke({"input": query})
 
        print(result["answer"])
        for doc in result["context"]:
            print("Source: ", doc.metadata["source"])
 
 
def build():
    w = ChatWebDoc()
    w.ingest([
        "https://www.webagesolutions.com/courses/WA3446-comprehensive-angular-programming",
        "https://www.webagesolutions.com/courses/AZ-1005-configuring-azure-virtual-desktop-for-the-enterprise",
        "https://www.webagesolutions.com/courses/AZ-305T00-designing-microsoft-azure-infrastructure-solutions",
        ])
 
def chat():
    w = ChatWebDoc()
 
    w.load()
 
    while True:
        query = input(">>> ")
 
        if len(query) == 0:
            continue
 
        if query == "/exit":
            break
         
        w.ask(query)

 
if __name__ in "__main__":
    if len(sys.argv) < 2:
        chat()
    elif sys.argv[1] == "--ingest":
        build()