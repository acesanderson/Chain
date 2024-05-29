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

embedding_models = {
    'ollama' : ['mxbai-embed-large, nomic-embed-text, all-minilm', 'snowflake-arctic-embed']
}

# client = chromadb.PersistentClient(path="../../vectordbs/ollama_both_libraries_chroma")
# collection = client.get_collection("langchain_store")

# def query_db(query, n_results=10):
#     q = collection.query(
#         query_texts=[query],
#         n_results=n_results,
#     )
#     return q['documents'][0]

### load "../exports/courselist_en_US.xlsx" as a pandas dataframe
df = pd.read_excel("../exports/courselist_en_US.xlsx")

### Get column names
print(df.columns)

['Locale', 'Course ID', 'Project ID', 'Course Name', 'Course Name EN',
'Activation Status', 'Display to Public', 'Display to QA',
'Course Description', 'Course Short Description', 'Content Type',
'Localization Type', 'Original Course Locale', 'Original Course ID',
'Equivalent English Course', 'Instructor ID', 'Instructor Name',
'Instructor Transliterated Name', 'Instructor Short Bio',
'Author Payment Category', 'Delivery Mode', 'Series End Date',
'Course Release Date', 'Course Updated Date', 'Course Archive Date',
'Course Retire Date', 'Replacement Course', 'Has Assessment',
'Has Challenge/Solution', 'LIL URL', 'Series', 'Limited Series',
'Manager Level', 'LI Level', 'LI Level EN', 'Sensitivity',
'Internal Library', 'Internal Subject', 'Primary Software',
'Media Type', 'Has CEU', 'Has Exercise Files', 'Visible Duration',
'Visible Video Count', 'Contract Type']

'Course Name EN'
'Course Description'
'Course Release Date'

### Remove all rows where the 'Course Release Date' is prior to 2018
df = df[df['Course Release Date'] > '2018-01-01']

### Reduce df to only two columns: 'Course Name EN', and 'Course Description'
df = df[['Course Name EN', 'Course Description']]

def convert_xlsx_to_df(file_path):
    """
    Load an Excel file as a pandas dataframe.
    """
    df = pd.read_excel(file_path)
    return df

def convert_course_list(df):
    """
    Convert a dataframe to a list of courses.
    """
    df = df[df['Course Release Date'] > '2018-01-01']
    df = df[['Course Name EN', 'Course Description']]
    return df


file_path = "../exports/courselist_en_US.xlsx"
df = convert_course_list(convert_xlsx_to_df(file_path))
df = df.fillna("nothing")



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

def wash_db(collection_name):
    """
    Delete all documents from a collection.
    """
    client.delete_collection(collection_name)

def create_db(collection_name):
    """
    Create a new collection.
    """
    client.create_collection(name=collection_name)

def wash():
    wash_db(collection_name)
    return create_db(collection_name)


