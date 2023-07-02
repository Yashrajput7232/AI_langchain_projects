import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from InstructorEmbedding import INSTRUCTOR




def pdf_to_text(pdfs):
    text=""
    for pdf in pdfs:
        pdfreader= PdfReader(pdf)
        for page in pdfreader.pages:
            text+= page.extract_text()
    return text

def break_into_chunks(text):
    text_spliter=CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks= text_spliter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    # model = INSTRUCTOR('hkunlp/instructor-xl')
    embeddings = OpenAIEmbeddings()
    # embeddings = model.encode([[text_chunks]])
    # embeddings = INSTRUCTOR(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore
def page_setup():
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    
    st.header("Upload Your PDF's and ask the questions 🤖")
    with st.sidebar:
        st.subheader("Your Documents")
        pdf_docs=st.file_uploader("Upload your Documents here" ,accept_multiple_files=True)
        if st.button("PROCESS"):
            with st.spinner("Processing"):
                raw_text=pdf_to_text(pdf_docs)
                chunks=break_into_chunks(raw_text)
                st.write(chunks)
                vectorstore = get_vectorstore(chunks)

                # create conversation chain
                # st.session_state.conversation = get_conversation_chain(
                #     vectorstore)
def main():
    load_dotenv()
    page_setup()
    
    

if __name__=="__main__":
    main()