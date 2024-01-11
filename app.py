import streamlit as st
from streamlit_option_menu import option_menu
from memory import memory_storage
from agent_chain import get_agent_chain
from default_text import default_text4
from generate_plot import generate_plot, retry_generate_plot
from markup import app_intro, how_use_intro
from modules import replace_default_dataset, save_uploaded_dataset
import pandas as pd
import os

if 'error' not in st.session_state:
    st.session_state['error'] = []

def tab1():

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("image.jpg", use_column_width=True)
    with col2:
        st.markdown(app_intro(), unsafe_allow_html=True)
    st.markdown(how_use_intro(),unsafe_allow_html=True) 


    github_link = '[<img src="https://badgen.net/badge/icon/github?icon=github&label">](https://github.com/ethanrom)'
    huggingface_link = '[<img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue">](https://huggingface.co/ethanrom)'

    st.write(github_link + '&nbsp;&nbsp;&nbsp;' + huggingface_link, unsafe_allow_html=True)
    
    st.markdown("<p style='font-size: 14px; color: #777;'>Disclaimer: This app is a proof-of-concept and may not be suitable for real-world decisions. During the Hackthon period usage information are being recorded using Langsmith</p>", unsafe_allow_html=True)



def tab2():

    dataset_option = st.radio("Select Dataset Option", ("Default", "Upload"))
    
    if dataset_option == "Default":
        if st.button("Use Default Dataset"):
            replace_default_dataset()
    else:
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded_file:
            save_uploaded_dataset(uploaded_file)

    st.header("🗣️ Chat")

    for i, msg in enumerate(memory_storage.messages):
        name = "user" if i % 2 == 0 else "assistant"
        st.chat_message(name).markdown(msg.content)

    if user_input := st.chat_input("User Input"):

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("Generating Response..."):

            with st.chat_message("assistant"):
                zeroshot_agent_chain = get_agent_chain()
                response = zeroshot_agent_chain({"input": user_input})

                answer = response['output']
                st.markdown(answer)

    
    if st.sidebar.button("Clear Chat History"):
        memory_storage.clear()


def tab3():

    dataset_option = st.radio("Select Dataset Option", ("Default", "Upload"))
    
    if dataset_option == "Default":
        if st.button("Use Default Dataset"):
            replace_default_dataset()
    else:
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded_file:
            save_uploaded_dataset(uploaded_file)

    st.header("📊 Data Visualization with NLP 🚀")
    st.markdown(
        """
        Explore your data like never before! Our Natural Language Processing (NLP)-powered tool transforms
        complex queries into stunning visualizations. Simply ask questions in plain language, and watch as
        insightful plots and charts come to life.
        """
    )

    question = st.text_area("🔍 Enter Your Query Here:", value=default_text4)

    result = None
    result2 = None

    if st.button("🚀 Generate Visualization"):


        with st.spinner('🤔 Thinking...'):
            result = generate_plot(question)
        st.session_state.generated_code = result

        st.subheader("👁️‍🗨️ Visualization")
        
        try:
            with st.spinner('📊 Generating Visualization...'):
                exec(st.session_state.generated_code)
        except Exception as e:
            st.error(f"Error executing generated code: {str(e)}")
            st.session_state.error = str(e)
            
        st.markdown("____")
        
        with st.expander("👩‍💻 View The Generated Code"):
            st.code(result, language="python")
                    

    if st.session_state.error and st.button("Retry"):
        with st.spinner('🤔 Thinking...'):
            result2 = retry_generate_plot(question, st.session_state.error, st.session_state.generated_code)
        st.session_state.generated_code2 = result2        
        st.subheader("👁️‍🗨️ Visualization")

        try:
            with st.spinner('📊 Generating Visualization...'):
                exec(st.session_state.generated_code2)            
        except Exception as e:
            st.error(f"Error executing generated code: {str(e)}")

        st.code(result2, language="python")  

def tab4():

    try:
        df = pd.read_csv("dataset.csv")

        st.header("Dataset Content")
        st.dataframe(df)

    except FileNotFoundError:
        st.error("File 'dataset.csv' not found in the current directory.")

    except pd.errors.EmptyDataError:
        st.error("File 'dataset.csv' is empty.")

    except pd.errors.ParserError:
        st.error("File 'dataset.csv' could not be parsed as a CSV file.")

def main():
    st.set_page_config(page_title="NaturalViz", page_icon=":memo:", layout="wide")

    #os.environ['LANGCHAIN_TRACING_V2'] = "true"
    #os.environ['LANGCHAIN_API_KEY'] == st.secrets['LANGCHAIN_API_KEY']

    tabs = ["Intro", "Data Visualization with NLP", "Chat", "View Dataset"]

    with st.sidebar:

        current_tab = option_menu("Select a Tab", tabs, menu_icon="cast")

    tab_functions = {
    "Intro": tab1,
    "Data Visualization with NLP": tab3,
    "Chat": tab2,
    "View Dataset": tab4,
    }

    if current_tab in tab_functions:
        tab_functions[current_tab]()

if __name__ == "__main__":
    main()