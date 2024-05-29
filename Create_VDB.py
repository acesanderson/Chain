"""
This module contains the class definition for VectorStore class.
VectorStore is a Retriever class that handles the storage and querying of vectors.

It handles:
### Initiating the VectorStore
- Text and CSV splitting
- Creating embeddings
- Storing vectors
### Querying the VectorStore
- Similarity search (and other ML functions)

Like a Tool, it can take a Parser object.
"""

from Chain import Chain, Model, Prompt, Parser
import pandas as pd
import chromadb
import ollama

# Embedding models here, though we are using chromadb default for now (all_minilm)
embedding_models = {
    'ollama' : ['mxbai-embed-large, nomic-embed-text, all-minilm', 'snowflake-arctic-embed']
}


def convert_xlsx_to_df(file_path):
    """
    Load an Excel file as a pandas dataframe.
    """
    df = pd.read_excel(file_path)
    return df

def convert_course_list(df):
    """
    Convert a dataframe to a list of courses. This is very specific to the use case of the Cosmo export.
    """
    ### remove all rows with a 'Course Release Date' before 2018
    df = df[df['Course Release Date'] > '2018-01-01']
    ### Remove all rows with "Activation Status" != "ACTIVE"
    df = df[df['Activation Status'] == "ACTIVE"]
    ### Remove all columns except "Course Name EN" and "Course Description"
    df = df[['Course Name EN', 'Course Description']]
    df = df.fillna("nothing")
    return df

file_path = "../exports/courselist_en_US.xlsx"
df = convert_course_list(convert_xlsx_to_df(file_path))

# initialized database
# embeddings_model = 'mxbai-embed-large'
# embeddings = ollama.Client(embeddings_model).embeddings
collection_name = "All_Courses-5-23-2024"
persist_directory = '../vectordbs/Library_RAG_Chain'
client = chromadb.PersistentClient(path=persist_directory)
collection = client.create_collection(name=collection_name)
collection = client.get_collection(name=collection_name)

# for row in df, add to db
for index, row in df.iterrows():
    print(f"Adding course {index} to db.")
    course_string = row[str('Course Name EN')] + ": " + row[str('Course Description')]
    collection.add(documents = [course_string], ids = [str(index)])

def query_db(query, n_results=10):
    q = collection.query(
        query_texts=[query],
        n_results=n_results,
    )
    return q['documents'][0]

