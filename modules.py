import streamlit as st
import pandas as pd
import os
import shutil

def replace_default_dataset():
    # Replace dataset.csv with dataset_backup.csv
    dataset_backup_path = "dataset_backup.csv"
    dataset_path = "dataset.csv"

    if os.path.exists(dataset_backup_path):
        shutil.copy(dataset_backup_path, dataset_path)
        st.success("Default dataset applied successfully.")
    else:
        st.warning("Default dataset backup not found.")

def save_uploaded_dataset(uploaded_file):
    # Save the uploaded dataset file and replace dataset.csv
    dataset_path = "dataset.csv"
    uploaded_file.seek(0)

    try:
        df = pd.read_csv(uploaded_file)
        df.to_csv(dataset_path, index=False)
        st.success("Dataset uploaded and applied successfully.")
    except pd.errors.EmptyDataError:
        st.warning("Uploaded dataset is empty.")
    except Exception as e:
        st.error(f"An error occurred: {e}")