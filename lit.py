# streamlit_app.py

import sys
sys.path.append(r"C:\Users\ASUS\Desktop\POC")  # Add the parent directory to the Python path

from test3 import create_vector_db, get_qa_chain
import streamlit as st

def main():
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

if __name__ == "__main__":
    main()
