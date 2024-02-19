# RAG based ChatBot

This is a RAG based ChatBot allows you to upload PDFs or website for retrival and talk to chatbot.

## Dependencies and Installation
------------

To install the chat App, please follow the steps:

1. Clone the repository to your local machine

2. Install the required dependencies by running the following command:
```
pip install -r requirements.txt
```

3. Obtain an API key from OpenAI and add it to the `.env` file in the project directory.
```commandline
OPENAI_API_KEY=your_secret_api_key
```

## Usage
------------

To chat with ChatBot, follow these steps:

1. Ensure that you have installed the required dependencies and added the OpenAI API key to the `.env` file.

2. Run the `main.py` file using the Streamlit CLI. Execute the following command:
```
streamlit run app.py
```

3. The application will launch in your default web browser.

4. Upload your PDFs or enter the website.

5. Ask questions using chat interface.