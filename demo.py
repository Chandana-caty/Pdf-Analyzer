# import streamlit as st
# from PyPDF2 import PdfReader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import os
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# import google.generativeai as genai
# from langchain.vectorstores import FAISS
# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain.chains.question_answering import load_qa_chain
# from langchain.prompts import PromptTemplate
# from dotenv import load_dotenv

# load_dotenv()
# os.getenv("GOOGLE_API_KEY")
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# def get_pdf_text(pdf_docs):
#     text=""
#     for pdf in pdf_docs:
#         pdf_reader= PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text+= page.extract_text()
#     return  text

# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
#     chunks = text_splitter.split_text(text)
#     return chunks

# def get_vector_store(text_chunks):
#     embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
#     vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
#     vector_store.save_local("faiss_index")  # Save the vector store locally

# def load_vector_store():
#     try:
#         embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
#         vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#         return vector_store
#     except Exception as e:
#         return None  # Return None if the vector store doesn't exist

# def get_conversational_chain():

#     prompt_template = """
#     Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
#     provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
#     Context:\n {context}?\n
#     Question: \n{question}\n
#     Answer:
#     """

#     model = ChatGoogleGenerativeAI(model = "models/chat-bison-001")

#     prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
#     chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

#     return chain

# def user_input(user_question):
#     embeddings = GoogleGenerativeAIEmbeddings(model = "models/text-embedding-004")
    
#     new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
#     docs = new_db.similarity_search(user_question)

#     chain = get_conversational_chain()

    
#     response = chain(
#         {"input_documents":docs, "question": user_question}
#         , return_only_outputs=True)

#     print(response)
#     st.write(user_question)
#     st.write(response["output_text"])

# def main():
#     st.set_page_config("Chat with PDFs", page_icon=":open_file_folder:") 
    
#     # Header
#     st.header(":rainbow[Chat with PDFs :material/docs:]")
    
#     # User Input
#     user_question = st.text_input(":orange[Ask a Question from the PDF Files :file_folder:]")
#     if user_question:
#         user_input(f'You asked: {user_question}')
        
#     # Sidebar
#     with st.sidebar:
#         st.subheader(":green[Upload Your Documents :material/docs:]")
#         st.title(":blue[Menu]")
#         pdf_docs = st.file_uploader(":red[Upload your PDF Files and Click on the Submit & Process Button]", accept_multiple_files=True)
#         if st.button("Submit & Process"):
#             with st.spinner("Processing..."):
#                 raw_text = get_pdf_text(pdf_docs)
#                 text_chunks = get_text_chunks(raw_text)
#                 # st.write(text_chunks)
#                 get_vector_store(text_chunks)
#                 st.success("Done")

# if __name__ == "__main__":
#     main()