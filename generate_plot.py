from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import pandas as pd
import os
import streamlit as st
from langchain_community.llms import HuggingFaceHub



os.environ["HF_TOKEN"] == st.secrets["HF_TOKEN"]


llm = HuggingFaceHub(
    huggingfacehub_api_token = os.environ["HF_TOKEN"],
    repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    task="text-generation",
    model_kwargs={
        "max_new_tokens": 512,
        "top_k": 50,
        "top_p": 0.1,
        "temperature": 0.2,
        "repetition_penalty": 1.03,
    },
)


def read_first_3_rows():
    dataset_path = "dataset.csv"
    try:
        df = pd.read_csv(dataset_path)
        first_3_rows = df.head(3).to_string(index=False)
    except FileNotFoundError:
        first_3_rows = "Error: Dataset file not found."

    return first_3_rows

def generate_plot(question):
    dataset_first_3_rows = read_first_3_rows()

    GENERATE_PLOT_TEMPLATE_PREFIX = """You are an expert in data visualization who can create suitable visualizations to find the required information. You have access to a dataset (dataset.csv) and you are gievn a question. Generate a python code with st.altair_chart to find the answer.
    First 3 rows of the dataset:"""

    DATASET = f"{dataset_first_3_rows}"

    GENERATE_PLOT_TEMPLATE_SUFIX = """
Question:
{question}

Example for protein count of different products:
import altair as alt
import pandas as pd
import streamlit as st

# Read the dataset
df = pd.read_csv('dataset.csv')

# Calculate the protein count of different products
product_protein = df.groupby('name')['protein'].sum().reset_index()

# Create the chart
chart = alt.Chart(product_protein).mark_bar().encode(
    x=alt.X('name:N', title='Product Name'),
    y=alt.Y('protein:Q', title='Protein Count')
)

# Display the chart
st.altair_chart(chart, use_container_width=True)

Generated Python Code:
"""

    template = GENERATE_PLOT_TEMPLATE_PREFIX + DATASET + GENERATE_PLOT_TEMPLATE_SUFIX
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    response = llm_chain.predict(question=question)
    return response



def retry_generate_plot(question, error_message, error_code):

    dataset_first_3_rows = read_first_3_rows()
    RETRY_TEMPLATE_PREFIX = """Current code attempts to create a visualization of dataset.csv to meet the objective. but it has encounted the given error. provide a corrected code. if you are adding comments or explanations they should start with #.

Example:
import altair as alt
import pandas as pd
import streamlit as st

# Read the dataset
df = pd.read_csv('dataset.csv')

# Calculate the total social media followers for each region
region_followers = df.groupby('Region of Focus')[['X (Twitter) Follower #', 'Facebook Follower #', 'Instagram Follower #', 'Threads Follower #', 'YouTube Subscriber #', 'TikTok Subscriber #']].sum().reset_index()

# Melt the dataframe to convert it into long format
region_followers = region_followers.melt(id_vars='Region of Focus', var_name='Social Media', value_name='Total Followers')

# Create the chart
chart = alt.Chart(region_followers).mark_bar().encode(
    x=alt.X('Region of Focus:N', title='Region of Focus'),
    y=alt.Y('Total Followers:Q', title='Total Followers'),
    color=alt.Color('Social Media:N', title='Social Media')
)

# Display the chart
st.altair_chart(chart, use_container_width=True)    

First 3 rows of the dataset:"""
    DATASET = f"{dataset_first_3_rows}"


    RETRY_TEMPLATE_SUFIX = """
Objective: {question}

Current Code:
{error_code}

Error Message:
{error_message}

Corrected Code:
"""

    retry_template = RETRY_TEMPLATE_PREFIX + DATASET + RETRY_TEMPLATE_SUFIX
    retry_prompt = PromptTemplate(template=retry_template, input_variables=["question", "error_message, error_code"])

    llm_chain = LLMChain(prompt=retry_prompt, llm=llm)
    response = llm_chain.predict(question=question, error_message=error_message, error_code=error_code)
    return response