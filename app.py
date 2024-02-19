import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv 
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import faiss
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from htmlTemplates import css, bot_template, user_template

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    
    return text

def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=200,
        chunk_overlap=40,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vectorstore_from_texts(text_chunks):
    embeddings = OpenAIEmbeddings()                # openai embeddings, fast but charge
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")    # instructor embeddings, slow but free
    vectorstore = faiss.FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_website_docs(website):
    loader = WebBaseLoader(website)
    return loader.load()

def get_vectorstore_from_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()                # openai embeddings, fast but charge
    vectorstore = faiss.FAISS.from_documents(documents, embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    retriever_chain = create_history_aware_retriever(llm, vectorstore.as_retriever(), prompt)

    doc_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}")
    ])
    document_chain = create_stuff_documents_chain(llm, doc_prompt)
    retrival_chain = create_retrieval_chain(retriever_chain, document_chain)
    return retrival_chain

def handle_user_input(user_input):
    response = st.session_state.conversation.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    st.session_state.chat_history.append(HumanMessage(user_input))
    st.session_state.chat_history.append(AIMessage(response["answer"]))
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)



def main():
    # loading secrets from dotenv
    load_dotenv()

    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    st.header("chat with virtual girlfriend :books:")
    user_input = st.text_input("Ask something to ChatBot")

    if user_input:
        handle_user_input(user_input)

    # initialization
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    

    with st.sidebar:
        st.subheader("Documents")
        pdf_docs = st.file_uploader("Upload your PDF files here and click 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get raw text
                raw_text = get_pdf_text(pdf_docs)

                # split text into chunks
                text_chunks = get_text_chunks(raw_text)

                # transform chunks into embeddings
                vectorstore = get_vectorstore_from_texts(text_chunks)

                # create conversation chain (retrival chain)
                st.session_state.conversation = get_conversation_chain(vectorstore)

        st.subheader("Website")
        website_input = st.text_input("Please enter the website url and click 'Fetch'")
        if st.button("Fetch"):
            with st.spinner("Fetching"):
                # get document from website
                website_docs = get_website_docs(website_input)

                # get vectorstore from documents
                vectorstore = get_vectorstore_from_docs(website_docs)

                # create conversation chain (retrival chain)
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == "__main__":
    main()