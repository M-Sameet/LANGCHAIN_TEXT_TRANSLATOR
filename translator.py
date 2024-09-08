import os
import streamlit as st
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if GOOGLE_API_KEY is None:
    raise ValueError("Google API key not found. Please set it in the .env file.")

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=GOOGLE_API_KEY, temperature=0)

def lang_translator(script, language):
    prompt_template_name = PromptTemplate(
        input_variables=['script', 'language'],
        template="I have a text.i need you to translate my text into {language}. My text is '{script}'"
    )
    name_chain = LLMChain(llm=llm, prompt=prompt_template_name)

    # rspnse
    response = name_chain.run({'script': script, 'language': language})

    return response

def main():
    st.markdown(
        """
        <style>

        .main-content {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
        }
       
        .stButton > button {
            color: white;
            background-color: #4CAF50;
            border-radius: 8px;
            border: none;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
        }
        .stButton > button:hover {
            background-color: #45a049;
        }
        .custom-bar {
            height: 4px;
            background-color: #4CAF50;
            margin: 20px 0;
        }
        </style>
        """, unsafe_allow_html=True
    )
    
    st.title("🌍 Language Translator")
    st.markdown('<div class="custom-bar"></div>', unsafe_allow_html=True)

    st.title("Language Translator")

    script = st.text_input("Enter your Text Here:") 
    languages = ["English", "Spanish", "French", "German", "Chinese", "Urdu", "Hindi", "Roman Urdu", "Hindi", "Japnese", "Russian", "Vietnamese", "Polish"]
    language = st.selectbox("Select the language in which you want to translate:", languages)

    if st.button("Translate"):
        if script and language:
            with st.spinner("Generating..."):
                response = lang_translator(script, language)
                st.write("Here is your script in selected language")
                st.write(response)
        else:
            st.warning("Please enter both the script and language.")
    st.markdown('<div class="custom-bar"></div>', unsafe_allow_html=True)        

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
