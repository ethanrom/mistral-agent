from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool
from langchain.chains import LLMMathChain
import streamlit as st
import pandas as pd
import plotly.express as px
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
        "top_p": 0.9,
        "temperature": 0.6,
        "repetition_penalty": 1.03,
    },
)

def csv_agnet(string):
    agent = create_csv_agent(
        llm,
        "dataset.csv",
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )

    ans = agent.invoke(string)
    return ans

#def csv_agnet(string):
#    agent = create_csv_agent(
#        ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613"),
#        "dataset.csv",
#        verbose=True,
#        agent_type=AgentType.OPENAI_FUNCTIONS,
#    )

#    ans = agent.run(string)
#    return ans

def math_tool(string):
    #llm = OpenAI(temperature=0)
    llm - llm
    llm_math_chain = LLMMathChain(llm=llm, verbose=True)
    res = llm_math_chain.run(string)
    return res

def load_data():
    df = pd.read_csv("dataset.csv", encoding="utf-8")
    return df

def plot_visualization(selected_option, x_column, y_column):
    df = load_data()

    if df.empty:
        return st.warning("The data is empty.")

    if x_column not in df.columns or y_column not in df.columns:
        return st.warning("Invalid columns selected.")

    if selected_option == "bar":
        fig = px.bar(df, x=x_column, y=y_column, title=f"{x_column} vs {y_column}")
    elif selected_option == "scatter":
        fig = px.scatter(df, x=x_column, y=y_column, title=f"{x_column} vs {y_column}")
    elif selected_option == "line":
        fig = px.line(df, x=x_column, y=y_column, title=f"{x_column} vs {y_column}")
    elif selected_option == "scatter_matrix":
        fig = px.scatter_matrix(df, dimensions=[x_column, y_column], title=f"Scatter Matrix: {x_column} vs {y_column}")
    elif selected_option == "box":
        fig = px.box(df, x=x_column, y=y_column, title=f"Box Plot: {x_column} vs {y_column}")
    elif selected_option == "heatmap":
        fig = px.imshow(df.pivot_table(index=x_column, columns=y_column, aggfunc='size').fillna(0),
                        labels=dict(x=x_column, y=y_column),
                        title=f"Heatmap: {x_column} vs {y_column}")
    else:
        return st.warning("Please select a valid plot type.")

    return st.plotly_chart(fig)


def parsing_input(string):
    selected_option, x_column, y_column = string.split(",")
    return plot_visualization(selected_option, x_column, y_column)


zeroshot_tools = [
    Tool(
        name="answer_qa",
        func=csv_agnet,
        description="Use this tool to query the dataset. input to this tool should be a standalone question. Include the correct row titles that are needed. Example Input format: How many rows are there in the dataset, which name has the highest calories",
        #return_direct=True,
    ),
    Tool(
        name="create_simple_plot",
        func=parsing_input,
        description="""Use this tool if the user asks to create x vs y plots. input must be a comma seperated list of: selected_option, x_column, y_column
        Example Inputs: 
        bar,calories,name

        Allowed options are: bar, line, scatter_matrix, box, heatmap
        you can decide plot type, x colllumn and y collumn based on the user input.
        """,
        #return_direct=True,
    ),
    Tool(
        name="Calculator",
        func=math_tool,
        description="useful when you need to do calculations. Example input: 21^0.43"
    ),
]
