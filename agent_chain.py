from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from memory import memory
from tools import zeroshot_tools
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
        "top_p": 0.2,
        "temperature": 0.1,
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


def get_agent_chain():


    dataset_first_3_rows = read_first_3_rows()

    prompt = PromptTemplate(

    input_variables = ['agent_scratchpad', 'chat_history', 'input', 'tool_names', 'tools'],
    template = ( f"""
You are a helpful assistant that can help users explore a dataset.
First 3 rows of the dataset:
{dataset_first_3_rows}
===="""
"""
TOOLS:
------
You has access to the following tools:

{tools}

To use a tool, please use the following format:

Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

Thought: Do I need to use a tool? No
Final Answer: [your response here]

Begin!

New input: {input}
{agent_scratchpad}"""
    )

    )


    conversational_agent_llm = llm
    #conversational_agent_llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=temperature, streaming=True)
    conversational_agent = create_react_agent(conversational_agent_llm, zeroshot_tools, prompt)
    room_selection_chain = AgentExecutor(agent=conversational_agent, tools=zeroshot_tools, verbose=True, memory=memory, handle_parsing_errors=True, max_iterations=4)
    return room_selection_chain