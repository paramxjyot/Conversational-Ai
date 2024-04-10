# Importing necessary libraries and modules
import neo4j
from langchain.vectorstores import FAISS
from neo4j import GraphDatabase
from langchain.llms import GooglePalm
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import GPT4AllEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
import sys

# Setting the standard output encoding to utf-8
sys.stdout.reconfigure(encoding='latin-1 ')

# Setting up API keys and initializing language models
api_key = "AIzaSyDZRGLrG6vlv0O3YbuA27tJrZIkR_L_52w"
llm_api_key = "AIzaSyDZRGLrG6vlv0O3YbuA27tJrZIkR_L_52w"  
llm = GooglePalm(google_api_key=api_key, temperature=0.7)
gpt4all_embd = GPT4AllEmbeddings()

# Neo4j database connection details
neo4j_uri = "neo4j+s://a4bd3e7e.databases.neo4j.io"  
neo4j_user = "neo4j"
neo4j_password = "aS-7et4yr5OH5b6jWUkZJg1r5rCuJPiQkT9DWPe6H8g"

# Initializing Neo4j driver
neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

# Function to create a vector database from a text file and populate Neo4j graph database
def create_vector_db():
    vectordb_file_path = "faiss_index"  
    
    # Reading texts from a text file
    with open(r"C:\Users\ASUS\Downloads\OUTPUT.txt", encoding="latin-1") as text_file:
        raw_text = text_file.read()
    
    # Splitting the raw text into chunks for processing
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len 
    )
    texts = text_splitter.split_text(raw_text)
    
    # Creating FAISS vector database from texts and GPT4All embeddings
    vectordb = FAISS.from_texts(texts, embedding=gpt4all_embd)

    # Saving vector database locally
    vectordb.save_local(vectordb_file_path)

    # Storing vector embeddings in the Neo4j graph database
    with neo4j_driver.session() as session:
        for i, embedding in enumerate(vectordb.embeddings):
            # Converting to a similar format to store them in the graph db
            embedding_str = str(embedding)
            
            # Creating nodes in Neo4j for each document with an ID and embedding
            session.run(
                "CREATE (n:Document {id: $id, embedding: $embedding})",
                {"id": i, "embedding": embedding_str}
            )

# Function to get a question-answering chain
def get_qa_chain(vectordb_file_path):
    # Loading FAISS vector database and GPT4All embeddings
    vectordb = FAISS.load_local(vectordb_file_path, gpt4all_embd)

    # Creating a retriever instance from the vector database
    retriever = vectordb.as_retriever(score_threshold=0.7)

    # Defining a prompt template for the question-answering task
    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    Try to give a 10 to 15 line answer for every question highliting the neccessary details.
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

# Running the code to create the vector database, get the QA chain, and perform a question-answering task
vectordb_file_path = "faiss_index"
create_vector_db()
chain, neo4j_embeddings = get_qa_chain(vectordb_file_path)
print(chain("what is bot penguin"))

# Closing the Neo4j driver connection
neo4j_driver.close()
