# COOP-Training

This project is part of the COOP Training program, focusing on LLMs based applications.


## Installation

1. Clone the repository:
   ```
   git clone https://github.com/OWDM/COOP-Training.git
   cd COOP-Training
   ```

2. (Optional but recommended) Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running the Arabic Summarizer Streamlit App

To run the Arabic Summarizer application:

1. Navigate to the Arabic Summarizer directory:
   ```
   cd week3/Arabic\ Summarizer
   ```

2. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

3. Open your web browser and go to the URL displayed in the terminal (usually `http://localhost:8501`).

The Streamlit app should now be running, and you can interact with the Arabic Summarizer through the web interface.

## Dependencies

This project relies on the following main packages:
- langchain and langchain-community
- openai
- faiss-cpu
- streamlit
- python-dotenv
- pydantic
- SQLAlchemy
- tiktoken

For a complete list with versions, see the `requirements.txt` file.


## Contact
