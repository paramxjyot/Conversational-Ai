from langchain.vectorstores import FAISS
from langchain.llms import GooglePalm
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import GPT4AllEmbeddings

api_key = "AIzaSyDZRGLrG6vlv0O3YbuA27tJrZIkR_L_52w"
llm = GooglePalm(google_api_key=api_key , temperature=0.7)

gpt4all_embd = GPT4AllEmbeddings()
vectordb_file_path = "faiss_index"

def create_vector_db():
    #loading the csv file 
    loader = CSVLoader(file_path = r"C:\Users\ASUS\Downloads\codebasics_faqs.csv" , source_column = "prompt")
    doc = loader.load()
    
    
    vectro_db = FAISS.from_documents(documents = doc , embedding = gpt4all_embd)   
    vectro_db.save_local(vectordb_file_path)
    
def get_qa_chain():
    # Load the vector database from the local folder
    vectordb = FAISS.load_local(vectordb_file_path, gpt4all_embd)

    # Create a retriever for querying the vector database
    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT})

    return chain

if __name__ == "__main__":
    create_vector_db()
    chain = get_qa_chain()
    print(chain("Do you have javascript course?"))