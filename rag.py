#### origianl blog: https://medium.com/@genuine.opinion/building-a-simple-rag-application-a-step-by-step-approach-a9b77fce04f9
import os
import getpass
import os.path
from os import listdir
#from dotenv import load_dotenv
from os.path import isfile, join
from typing import Literal, get_args
#from langchain.chains import RetrievalQA
#from langchain_openai import ChatOpenAI,OpenAIEmbeddings

# loading API keys from env
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# loading model and defining embedding
llm = ChatOpenAI(temperature=0, model='gpt-3.5-turbo')
embeddings = OpenAIEmbeddings()
#case1 use wikipedia as the external source
Wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
data = Wikipedia.run(query)
split_docs = [Document(page_content=sent) for sent in data.split('\n')]
data_set = DocArrayInMemorySearch.from_documents(documents = split_docs, embedding = embeddings)

#case2: RAG using a pdf file
# fragmenting the document content to fit in the number of token limitations
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)

# get files from the target directory
my_file = [f for f in listdir(target_folder) if isfile(join(target_folder, f))]
my_file = target_folder + my_file[0]

# load uploaded pdf file
loader = PyPDFLoader(my_file)
data = loader.load()
split_docs = text_splitter.split_documents(data)

data_set = DocArrayInMemorySearch.from_documents(documents = split_docs, embedding = embeddings)

qa = RetrievalQA.from_chain_type(
llm = llm,
      chain_type="stuff",
      retriever = data_set.as_retriever(), # repla
      verbose=True
)


import os, os.path
import shutil
import pathlib
import streamlit as st

st.title("RAG Model Benefit Demonstration")
st.header("\n A Simplified Approach")

if 'selection' not in st.session_state:
    st.session_state.selection = ""
# activating selection box for user choice 
options = False
emp = st.empty()
vari = emp.selectbox(
    key = "Options",
    label = "Please select the option for query running:",
    options = ("Wikipedia", "Research Paper")
)

wiki_query = st.text_input(
label = "Please input your Wikipedia search query",
      max_chars = 256
   )
# if st.button("Submit"):
#     return wiki_query


with st.form(key="doc_upload", clear_on_submit=False):
	uploaded_doc = st.file_uploader(
	      label="Please upload your document",
	      accept_multiple_files = False,
	      type=['pdf']
	      )
     
research_query = st.text_input(
label = "Please input what you want to search",
max_chars = 256
        )
submit_button1 = st.form_submit_button("Load Document")

if submit_button1:
	with open(os.path.join(target_folder, uploaded_doc.name), 'wb') as f:
	      f.write(uploaded_doc.getbuffer())