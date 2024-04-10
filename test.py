# Importing necessary libraries and modules
import neo4j
from langchain.vectorstores import FAISS
from neo4j import GraphDatabase
from langchain.llms import GooglePalm
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import GPT4AllEmbeddings
import streamlit as st 

# Setting up API keys
api_key = "AIzaSyDZRGLrG6vlv0O3YbuA27tJrZIkR_L_52w"
llm_api_key = "AIzaSyDZRGLrG6vlv0O3YbuA27tJrZIkR_L_52w"  

# Initializing language models
llm = GooglePalm(google_api_key=api_key, temperature=0.7)

#embedding model 
gpt4all_embd = GPT4AllEmbeddings()

# Neo4j database connection details
neo4j_uri = "neo4j+s://a4bd3e7e.databases.neo4j.io"  
neo4j_user = "neo4j"
neo4j_password = "aS-7et4yr5OH5b6jWUkZJg1r5rCuJPiQkT9DWPe6H8g"

# Initializing Neo4j driver
neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

# Function to create a vector database and populate Neo4j graph database
def create_vector_db():
    vectordb_file_path = "faiss_index"  
    loader = CSVLoader(file_path=r"C:\Users\ASUS\Downloads\codebasics_faqs.csv", source_column="prompt")
    doc = loader.load()

    # Creating FAISS vector database from documents and GPT4All embeddings
    vectordb = FAISS.from_documents(documents=doc, embedding=gpt4all_embd)

    # Saving vector database locally
    vectordb.save_local(vectordb_file_path)

    # Storing vector embeddings in the Neo4j graph database
    with neo4j_driver.session() as session:
        for i, embedding in enumerate(vectordb.embeddings):
            # Converting embedding to a string for storage in Neo4j
            embedding_str = str(embedding)
            
            # Creating nodes in Neo4j for each document with an ID and embedding
            session.run(
                "CREATE (n:Document {id: $id, embedding: $embedding})",
                {"id": i, "embedding": embedding_str}
            )

# Function to create a retrieval question-answering chain
def get_qa_chain(vectordb_file_path):
    # Loading FAISS vector database and GPT4All embeddings
    vectordb = FAISS.load_local(vectordb_file_path, gpt4all_embd)

    # Creating a retriever instance from the vector database
    retriever = vectordb.as_retriever(score_threshold=0.7)

    # Defining a prompt template for the question-answering task
    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    # Creating a PromptTemplate object
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Creating a RetrievalQA chain with specified parameters
    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT})

    # Querying Neo4j graph database for data
    with neo4j_driver.session() as session:
        result = session.run("MATCH (n:Document) RETURN n.embedding AS embedding")
        neo4j_embeddings = [record["embedding"] for record in result]

    return chain, neo4j_embeddings

# File path for the vector database
vectordb_file_path = "faiss_index"

# Creating the vector database and obtaining the question-answering chain
create_vector_db()
chain, neo4j_embeddings = get_qa_chain(vectordb_file_path)

# Performing a question-answering task and printing the result
print(chain("do you have power bi courses"))
st.title("ASK YOUR QUESTION")

    # Button to create the knowledge base
btn_create_kb = st.button("Create Knowledgebase")
if btn_create_kb:
    create_vector_db()

    # Input field for the user to enter a question
question = st.text_input("Question:")

if question:
        # Getting the QA chain
    chain, _ = get_qa_chain("faiss_index")

        # Obtaining the response
    response = chain(question)

        # Displaying the answer
    st.header("Answer")
    st.write(response["result"])


