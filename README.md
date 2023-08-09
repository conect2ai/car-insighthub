# Car InsightHub 🚗💡

Car InsightHub is a platform designed to provide detailed insights and knowledge about automotive manuals. Using cutting-edge technology, you can process and understand PDFs containing information about cars and answer questions related to these data.

## 🎯 Main Features

1. **PDF Processing**: Upload your PDF files with the specific naming format Brand_Model_Year.pdf. This naming convention allows the application to recognize the brand, model, and year of the car from the filename itself
2. **Insights Extraction**: Extract important information like brand, model, and year directly from PDF file names
3. **Smart Querying**: Ask questions about the loaded manuals and get instant answers
4. **Chat Export**: Export the chat with the bot to a PDF file with a custom layout
5. **Filtering by Brand, Model, and Year**: Filter documents by brand, model, and year for more accurate answers

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- Streamlit
- OpenAI
- PyPDF2
- Other dependencies listed in the `requirements.txt` file

### Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/conect2ai/car-insighthub.git
cd car-insighthub
pip install -r requirements.txt
```

### Execution

Run the Streamlit application:

```bash
streamlit run app.py
```

The application will run in the browser at `http://localhost:8501`.

## 📝 Usage

- Enter your OpenAI API key
- Upload the PDFs of car manuals
- Process the PDFs by clicking the "Process PDFs" button
- Select the brand, model, and year to filter the documents
- Ask questions related to the uploaded manuals
- Export the chat with the bot if desired

You need to have an OpenAI API key. If you don't have one, you can sign up for an account at [OpenAI](https://openai.com/) and obtain an API key from the dashboard.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.