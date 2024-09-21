import streamlit as st
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_google_genai.chat_models import chatGoogleGenerativeAI
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
import pandas as pd
import os
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from html_temps import css, bot_template,user_template

def get_text(pdfs):
    text = ""
    for pdf in pdfs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n",chunk_size = 1000,chunk_overlap = 200,length_function=len)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):     
    # Create embeddings using a Google Generative AI model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Create a vector store using FAISS from the provided text chunks and embeddings
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

    # Save the vector store locally with the name "faiss_index"
    # vector_store.save_local("faiss_index")

    return vector_store

def get_conv_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model = "gemini-1.5-pro")
    memory = ConversationBufferMemory(memory_key='chat_history',return_messages = True)
    conv_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conv_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    st.set_page_config(page_title="LLM PDF's",page_icon=":books:")

    st.write(css,unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.header("Ask whatever you want from PDF's")
    user_question = st.text_input("Ask a question from your documents:")
    if user_question:
        handle_userinput(user_question)

    st.write(user_template.replace("{{MSG}}","W'sap bot"),unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}","W'sap man"),unsafe_allow_html=True)

    

    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs = st.file_uploader("Upload only PDF's(OCR)",accept_multiple_files=True)
        if st.button("Infer"):
            with st.spinner("Infering"): 
                #get pdf text
                raw_text = get_text(pdf_docs)

                #get text chunks
                
                text_chunks = get_chunks(raw_text)

                #create vector store
                
                vectorstore = get_vector_store(text_chunks)

                #creating conversation chain

                st.session_state.conversation = get_conv_chain(vectorstore)

    st.session_state.conversation





if __name__=="__main__":
    main()