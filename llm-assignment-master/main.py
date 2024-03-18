import os
# import PIL
# from PyPDF2 import PdfReader
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import google.generativeai as genai
# from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
# from langchain_core.messages import HumanMessage
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import pymongo

# from langchain.docstore.document import Document

# from langchain_google_genai import ChatGoogleGenerativeAI


#os.getenv("GOOGLE_API_KEY")
genai.configure(api_key= "AIzaSyBmZtXjJgp7yIAo9joNCZGSxK9PbGMcVaA")

raw_text = ''

# read all pdf files and return text

llm = ChatGoogleGenerativeAI(model="gemini-pro-vision")


# llm = genai.GenerativeModel('gemini-pro-vision')

def get_pdf_text(pdf_docs, i):
    text = ""

    pdf_reader = PdfReader(pdf_docs)
    for page in pdf_reader.pages:
        text += page.extract_text()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text=text)

    client = pymongo.MongoClient()

    db = client.my_database

    collection = db.my_collection

    document = {"text": ''.join(chunks)}

    collection.insert_one(document)

    client.close()

    return ''.join(chunks)


def get_conversational_chain():
    prompt_template = """
    you are given the following context, make sure to provide all the details.\n\n
    Context:\n""" + st.session_state['raw_text'] + """ History:{history}\nHuman:{input} \n\n Ai Assistant:"""

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   client=genai,
                                   temperature=0.3,
                                   )
    prompt = PromptTemplate(template=prompt_template, input_variables=['history', 'input'])
    chain = ConversationChain(llm=model, prompt=prompt, memory=ConversationBufferMemory(ai_prefix="AI Assistant"))
    return chain


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "upload some pdfs and ask me a question"}]


def user_input(user_question):
    # new_db = FAISS.load_local("faiss_index", embeddings)
    # docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    # print(raw_text)
    response = chain.predict(input=user_question)

    print(response)
    return response


def main():
    global raw_text
    st.set_page_config(
        page_title="PDF Chatbot",
        page_icon="🤖"
    )

    # Sidebar for uploading PDF files
    with st.sidebar:
        pdf_docs = st.file_uploader(
            "Upload your Resume and Click on the Submit & Process Button", accept_multiple_files=False)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                upload_urls = pdf_docs
                raw_text = get_pdf_text(upload_urls, 0)
                # print(raw_text)
                st.session_state['raw_text'] = raw_text
                st.success("Done")

    # Main content area for displaying chat messages
    st.title("Chat with Documents")
    # st.write("Welcome to the chat!")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    # Chat input
    # Placeholder for chat messages

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "Upload Your Document to get started"}]

    if "raw_text" not in st.session_state.keys():
        st.session_state['raw_text'] = ''

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Display chat messages and bot response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = user_input(prompt)
                placeholder = st.empty()
                placeholder.markdown(response)
        if response is not None:
            message = {"role": "assistant", "content": response}
            st.session_state.messages.append(message)


if _name_ == "_main_":
    main()
