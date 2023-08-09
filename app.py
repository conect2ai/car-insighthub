import os
import streamlit as st
from streamlit_chat import message
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from collections import deque
from io import BytesIO
import base64
from langchain.schema import Document

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.units import inch
from reportlab.platypus.flowables import Image
from PIL import Image as PilImage
from reportlab.platypus import Spacer


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'pdfs_processed' not in st.session_state:
        st.session_state.pdfs_processed = False
    if 'knowledge_base' not in st.session_state:
        st.session_state.knowledge_base = None
    if 'messages' not in st.session_state:
        st.session_state.messages = deque(maxlen=6)  # Only keep the 10 most recent messages
    if 'all_messages' not in st.session_state:  # create a new list for all messages
        st.session_state.all_messages = []

def extract_metadata_from_filename(filename):
    marca, modelo, ano = filename.rstrip('.pdf').split('_')
    return dict(marca=marca, modelo=modelo, ano=ano)

def process_pdfs(pdfs):
    """Extract text from PDFs and create a knowledge base."""
    with st.spinner('Processing PDFs...'):
        list_of_documents = []
        marcas, modelos, anos = set(), set(), set()

        for pdf in pdfs:
            text = ''
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()

            filename = pdf.name
            metadata = extract_metadata_from_filename(filename) # Extrai os metadados do nome do arquivo

            marcas.add(metadata['marca'])
            modelos.add(metadata['modelo'])
            anos.add(metadata['ano'])

            text_splitter = CharacterTextSplitter(
                separator='\n',
                chunk_size=700,
                chunk_overlap=35,
                length_function=len
            )

            chunks = text_splitter.split_text(text)

            # Create embeddings
            embeddings = OpenAIEmbeddings(chunk_size=500)

            # Create metadata for each chunk
            metadata_chunks = [metadata for _ in chunks]

            # Create documents with content and metadata
            for chunk, meta in zip(chunks, metadata_chunks):
                list_of_documents.append(Document(page_content=chunk, metadata=meta))

        # Create FAISS index
        st.session_state.knowledge_base = FAISS.from_documents(list_of_documents, embeddings)
        st.session_state.marcas = list(marcas)
        st.session_state.modelos = list(modelos)
        st.session_state.anos = list(anos)
        st.session_state.pdfs_processed = True
        st.success('PDFs processed. You may now ask questions.')


def answer_question(question, marca, modelo, ano):
    """Generate an answer to a question using the knowledge base."""
    st.session_state.messages.append({'message': question, 'is_user': True})  # Add user question to the message list
    st.session_state.all_messages.append({'message': question, 'is_user': True})  # Add user question to the all_messages list

    with st.spinner('Thinking...'):
        results_with_scores = st.session_state.knowledge_base.similarity_search_with_score(question, filter=dict(marca=marca, modelo=modelo, ano=ano))
        docs = [doc for doc, score in results_with_scores]  # Convert to a list of documents

        llm = OpenAI(model_name='gpt-4')
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=question)
            print(cb)

    st.session_state.messages.append({'message': response, 'is_user': False})  # Add bot response to the message list
    st.session_state.all_messages.append({'message': response, 'is_user': False})  # Add bot response to the all_messages list

def on_page(canvas, doc):
    # This function will be called for each page during the PDF creation process.
    # It receives a `canvas` object that can be used to draw on the page,
    # and a `doc` object that contains information about the document.
    
    # Add your image file
    img_path = './img/logo.png'
    # Load your image file with PIL
    pil_image = PilImage.open(img_path)

    # Get the original width and height of the image
    orig_width, orig_height = pil_image.size

    # Define the width you want for the image in the PDF
    img_width = 1.0 * inch

    # Calculate the height based on the original image's aspect ratio
    img_height = img_width * orig_height / orig_width

    img = Image(img_path, width=img_width, height=img_height)
    
    # Draw image at the top of the page
    x_position = 1.09 * inch  # adjust as necessary
    img.drawOn(canvas, x_position, doc.height + 1 * inch)  # adjust second and third parameter as necessary

def export_chat_to_pdf():
    buffer = BytesIO()

    doc = SimpleDocTemplate(buffer, pagesize=letter)
    
    story = []
    styles = getSampleStyleSheet()
    style = styles['BodyText']
    style.alignment = 4  # Justify text

    # Add a space after the image
    story.append(Spacer(1, 0.5*inch))  # adjust the second parameter as necessary

    # Add chat messages in pairs, separated by a Spacer
    for i in range(0, len(st.session_state.all_messages), 2):
        user_msg = st.session_state.all_messages[i]
        bot_msg = st.session_state.all_messages[i+1] if i + 1 < len(st.session_state.all_messages) else None

        user_text = 'You: ' + user_msg['message']
        para = Paragraph(user_text, style)
        story.append(para)

        if bot_msg:
            bot_text = 'Bot: ' + bot_msg['message']
            para = Paragraph(bot_text, style)
            story.append(para)

        # Add a Spacer after each user-bot pair
        story.append(Spacer(1, 0.2*inch))  # Adjust the second parameter to control the space height

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)  # The function `on_page` will be called for each page

    pdf_bytes = buffer.getvalue()
    buffer.close()

    return pdf_bytes


def display_chat():
    """Display the chat messages."""
    for i, msg in enumerate(st.session_state.messages):
        message(msg['message'], is_user=msg['is_user'], key=str(i))

def main():
    initialize_session_state()
    st.header('Car InsightHub')

    openai_key = st.text_input('Enter your OpenAI API key:', type='password', key='openai_key')


    if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key
            st.success('API key provided', icon='✅')
    else:
        st.warning('Please enter your OpenAI API key.', icon='⚠️')
        st.stop()

    pdfs = st.file_uploader('Choose PDF files', type=['pdf'], accept_multiple_files=True)

    # Add a button to confirm PDF processing
    process_pdfs_button = st.button('Process PDFs')

    # Extract text from the uploaded PDF
    if process_pdfs_button and pdfs:
        process_pdfs(pdfs)

    if st.session_state.pdfs_processed:
        # Criar um contêiner para os widgets
        filters_container = st.container()

        with filters_container:
            selected_marca = st.selectbox('Choose a brand:', st.session_state.marcas)
            selected_modelo = st.selectbox('Choose a model:', st.session_state.modelos)
            selected_ano = st.selectbox('Choose a year:', st.session_state.anos)
        
        # Placeholder for question input
        question_placeholder = st.empty()
        user_question = question_placeholder.text_input('Ask a question about your PDF:', key='text_input')
        
        submit_question_button = st.button('Submit Question') 

        if submit_question_button and user_question and st.session_state.pdfs_processed:  # Check if submit button is pressed
            answer_question(user_question, selected_marca, selected_modelo, selected_ano)
            display_chat()
        
        export_chat_button = st.button('Export Chat')

        if len(st.session_state.all_messages) > 0:  # Display the export button only if there's at least one message
            if export_chat_button:

                # Generate PDF bytes
                pdf_bytes = export_chat_to_pdf()

                # Create a download link
                b64 = base64.b64encode(pdf_bytes).decode()
                linko= f'<a href="data:application/octet-stream;base64,{b64}" download="chat_history.pdf">Click Here to download your PDF file</a>'
                st.markdown(linko, unsafe_allow_html=True)

if __name__ == '__main__':
    main()