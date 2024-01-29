import openai
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain_community.chat_models import ChatOpenAI

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")


# Load documents
@st.cache_data
def setup_documents(pdf_file_path, chunk_size, chunk_overlap):
    loader = PyPDFLoader(pdf_file_path)
    docs_raw = loader.load()
    docs_raw_text = [doc.page_content for doc in docs_raw]
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    docs = text_splitter.create_documents(docs_raw_text)

    return docs


# summarize documents
def custom_summary(docs, llm, custom_prompt, chain_type, num_summaries):
    custom_prompt = custom_prompt + """:\n{text}"""
    COMBINE_PROMPT = PromptTemplate(template=custom_prompt, input_variables=["text"])
    MAP_PROMPT = PromptTemplate(
        template="""Summarize: \n{text}""", input_variables=["text"]
    )

    if chain_type == "map_reduce":
        chain = load_summarize_chain(
            llm,
            chain_type=chain_type,
            map_prompt=MAP_PROMPT,
            combine_prompt=COMBINE_PROMPT,
        )

    else:
        chain = load_summarize_chain(llm, chain_type=chain_type)

    summaries = []
    for i in range(num_summaries):
        summary_output = chain({"input_documents": docs}, return_only_outputs=True)
        summaries.append(summary_output)

    return summaries


# chunk documents
def color_chuncks():
    pass


def main():
    st.set_page_config(layout="wide")
    st.title("SummarEase.Ai")

    llm = st.sidebar.selectbox("Select Language Model", ["GPT-3.5", "GPT-4"])

    chain_type = st.sidebar.selectbox(
        "Select Chain Type", ["map_reduce", "stuff", "redefine"]
    )

    chunk_size = st.sidebar.slider(
        "Select Chunk Size", min_value=20, max_value=10000, step=10, value=2000
    )
    chunk_overlap = st.sidebar.slider(
        "Select Chunk Overlap", min_value=5, max_value=5000, step=10, value=200
    )

    if st.sidebar.checkbox("Debug Chunk Size"):
        pass
    else:
        user_prompt = st.text_input("What would you like to summarize today?")
        pdf_file_path = st.file_uploader("Upload a PDF file", type=["pdf"])

        temperature = st.sidebar.number_input(
            "Set the GPT Temperature",
            min_value=0.0,
            max_value=1.0,
            step=0.1,
            value=0.5,
        )
        num_summaries = st.sidebar.number_input(
            "How many summaries would you like?",
            min_value=1,
            max_value=10,
            value=1,
            step=1,
        )


if __name__ == "__main__":
    main()
