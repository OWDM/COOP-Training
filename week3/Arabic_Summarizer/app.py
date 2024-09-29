import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain
from langchain.schema import SystemMessage
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import os
from typing import Dict
import time

# Set your OpenAI API key here
os.environ["OPENAI_API_KEY"] = "sk-pr..."

def create_vectorstore(text: str) -> FAISS:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(text)
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(texts, embeddings)

def extract_key_info(article: str) -> Dict[str, str]:
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
    vectorstore = create_vectorstore(article)
    retriever = vectorstore.as_retriever()

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )

    questions = [
        "What are the main technical concepts discussed in this article?",
        "What are the key findings or advancements mentioned?",
        "Are there any specific companies or researchers mentioned?",
        "What potential impacts or applications are discussed?"
    ]

    results = {}
    for question in questions:
        results[question] = qa_chain.run(question)

    return results

def generate_summary(article: str, key_info: Dict[str, str]) -> str:
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o")
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are an expert in summarizing technical news articles. Your task is to create a concise and structured summary of the given article, focusing on the key technical information."""),
        HumanMessagePromptTemplate.from_template("""
Article: {article}

Key Information:
{key_info}

Please provide a structured summary of this article, following these guidelines:

1. List all relevant statistics mentioned in the article with their descriptions to help you understand the content.

2. Then, provide a concise summary of the article using the following structure:
   a. Determine the primary category for the article from the following options:
       - Organizations: For articles about non-governmental organizations, international bodies, or institutions.
       - Governments: For articles about governmental actions, policies, or initiatives.
       - Universities: For articles about research, scientific findings, or academic reports.
       - Companies: For articles about businesses, corporations, or commercial entities.
    Choose the category that best fits the main focus of the article.
   b. Focus on the actual title for the article and then adjust it to start with a noun or an entity relevant to the news.
   c. Then, provide a summary that:
      - First Sentence: Describes what was developed or achieved.
      - Second Sentence: Briefly explains the functionality or purpose of the development.
      - Third Sentence: Mentions key results or findings.
      - Fourth Sentence: Provides any future plans or goals related to the development.
   d. Keep the summary under 180 words, excluding the category and the title.
   e. Focus on clarity and conciseness.
   f. Focus on the important number and mention it if it is important..
Format your response as follows:
                                                 
[Category]
                                                 
[Title]

                                                                                          
[Four-sentence summary]

                                       
Do not include any labels like "Title:" or "Summary:" or "Category". Start directly with the Category, followed by a line break, and the title, followed by a line break, and then, finally with the summary.
""")
    ])
    
    chain = LLMChain(llm=llm, prompt=prompt)
    summary = chain.run(article=article, key_info=str(key_info))
    return summary

def translate_to_arabic(text: str) -> str:
    llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini",verbose=True)
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are an expert translator specializing in technical translations from English to Arabic. Your task is to provide an accurate and fluent translation that preserves the technical nuances and structure of the original text."""),
        HumanMessagePromptTemplate.from_template("""
Translate the following text to Arabic, but transliterate all proper nouns and brand names, ensuring they are written in Arabic script based on their pronunciation, without translating their meaning. For the first occurrence of each proper noun or brand name, mention the English version between two brackets. For example, write جوجل (Google) the first time, but in subsequent mentions, only write جوجل. After the first mention, use only the Arabic transliteration without the English in brackets
If the there a  percentage or numerical statistic, enclose it in parentheses. For example, change '30%' to '(30%)'. 
{text}

Please ensure that the translation follows the same format as the original, with a Category on the first line (do not add ال to the word, make it indefinite), followed by the title and the summary. Do not add any labels in Arabic for "Title" or "Summary" or "Category".
""")
    ])
    
    chain = LLMChain(llm=llm, prompt=prompt)
    translation = chain.run(text=text)
    return translation

def process_article(article: str) -> str:
    key_info = extract_key_info(article)
    summary = generate_summary(article, key_info)
    arabic_summary = translate_to_arabic(summary)
    return arabic_summary

st.markdown("""
<style>
    .rtl {
        direction: rtl;
        text-align: right;
        font-family: Arial, sans-serif;
        background-color: #262730;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

st.title("Arabic Summarizer")

article = st.text_area("Enter your text here:", height=300)

if st.button("Generate a Summary"):
    if article:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process the article
        try:
            # Extracting key information
            status_text.text("Reading ... 📚")
            progress_bar.progress(25)
            key_info = extract_key_info(article)
            time.sleep(0.5)  # Simulate processing time
            
            # Generating summary
            status_text.text("Crafting the summary... ✍️")
            progress_bar.progress(50)
            summary = generate_summary(article, key_info)
            time.sleep(0.5)  # Simulate processing time
            
            # Translating to Arabic
            status_text.text("Translating to Arabic...🔄")
            progress_bar.progress(75)
            arabic_summary = translate_to_arabic(summary)
            time.sleep(0.5)  # Simulate processing time
            
            # Complete
            progress_bar.progress(100)
            status_text.text("Your Arabic summary is ready! ✨")
            time.sleep(0.5)  # Pause to show completion
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Display results
            st.success("Article processed successfully!")
            st.subheader("Arabic Summary:")
            st.markdown(f'<div class="rtl">{arabic_summary}</div>', unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter an article to process.")