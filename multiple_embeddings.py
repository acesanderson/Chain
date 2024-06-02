"""
This is a test runner for various embeddings for course deescriptions and transcripts.

Purpose:
- Find the best quality for VDBs returning course suggestions for a given query.
- Compare the quality of the embeddings for the different VDBs.
- Identify which data is actually best for Library (is it short descriptions, long descriptions, or transcripts?)

How I'll build this:
- [ ] come up with different types of queries
- [ ] identify my embeddings I'll test (chroma default, what else?)
- [ ] create the VDBs (by amending the three scripts in data folder: transcripts.py, short_descriptions.py, long_descriptions.py)
- [ ] create a test runner that will run the queries against the VDBs and return the results
- [ ] results should be some complex json
- [ ] pretty print the results (as a set of csvs? or somethihg)
- [ ] manually evaluate the quality of the results
"""

from time import time
import ollama
import chromadb
import json         # for pretty printing dicts
import pandas as pd

## Load short course descriptions into memory
## This is from data/short_descriptions_db.py
excel_file = '../data/exports/courselist_en_US.xlsx'
df = pd.read_excel(excel_file)
df = df.astype(str).fillna('')\
active_courses = df[(df['Activation Status'] == 'ACTIVE') &
                    (df['Course Release Date'] > '2018-01-01') &
                    (df['Locale'] == 'en_US')]
# get only these two columns: "Course Name EN" and "Course Short Description"
# handle utf encoding
def clean_text(text):
    replacements = {
        'â€™': "'",
        'â€œ': '"',
        'â€': '"',
        'â€˜': "'",
        'â€”': '—',  # Em-dash
        'â€“': '–'   # En-dash
    }
    for wrong, right in replacements.items():
        text = text.replace(wrong, right)
    return text

# clean the text, save as tuples (name, description)
short_descriptions = []
for index, row in active_courses.iterrows():
    clean_name = clean_text(row['Course Name EN'])
    clean_description = clean_text(row['Course Short Description'])
    short_descriptions.append((clean_name, clean_description))

# default model in chroma, which we've already used, is all-minilm
ollama_models = """mxbai-embed-large
nomic-embed-text
avr/sfr-embedding-mistral
hellord/e5-mistral-7b-instruct:Q4_0
snowflake-arctic-embed
snowflake-arctic-embed:22m""".split('\n')

# create the database we will be loading our various embeddings into
embeddings_test_directory = '/home/bianders/Brian_Code/Chain_Framework/data/vectordbs/Embeddings_Test'
embeddings_test_client = chromadb.PersistentClient(path=embeddings_test_directory)

short_descriptions_collection_template = "Short_Descriptions_5_23_2024_"

###!!!!!!!!!
### YOU NEED TO CREATE EMBEDDING FUNCTIONS, THIS IS JUST USING CHROMA DEFAULT
###!!!!!!!!!

def create_short_descriptions_collection_for_model(short_descriptions, model):
    short_descriptions_collection_name = short_descriptions_collection_template + model
    embeddings_test_client.create_collection(name=short_descriptions_collection_name)
    short_descriptions_collection = embeddings_test_client.get_collection(name=short_descriptions_collection_name)
    start = time()
    for index, short_description in enumerate(short_descriptions):
        short_descriptions_collection.add(
            documents=[short_description[0] + "::" + short_description[1]],
            ids=[short_description[0]]
        )
        print(f"Added document {index + 1} of {len(short_descriptions)} to the database for model {model}.")
    end = time()
    return short_descriptions_collection_name, end - start

# our function for querying the stuff
def query_short_descriptions(query, n_results=10, collection = ""):
    """
    This currently returns the first document in the query results.
    You can imagine some use cases where you want the ids.
    In future, that can be iomplemented if necessary by changing 'documents' to 'ids' in the return.
    """
    collection = embeddings_test_client.get_collection(name=collection)
    q = collection.query(
        query_texts=[query],
        n_results=n_results,
    )
    return q['documents'][0]

if __name__ == "__main__":
    results = {}
    for model in ollama_models:
        collection = create_short_descriptions_collection_for_model(short_descriptions, model)
        print(f"Model: {model} took {collection[1]} seconds to load the short descriptions.")
        results[model]['collection'] = collection[0]
        results[model]['duration'] = collection[1]
    
    print (json.dumps(results, indent=4))


# query_short_descriptions('I want to learn python', collection="Short_Descriptions_5_23_2024_mxbai-embed-large")

# our test queries

# from chromadb import Documents, EmbeddingFunction, Embeddings
# class MyEmbeddingFunction(EmbeddingFunction):
#     def __call__(self, input: Documents) -> Embeddings:
#         # embed the documents somehow
#         return embeddings



# queries = [
#     "The Six Morning Habits of High Performers"
#     "I want to learn The Six Morning Habits of High Performers"
#     "I want to learn Python programming."
#     "I want to start a career in Sales."
#     "I'm creating a startup and need to learn basic digital marketing techniquesl."
#     "I have been hired to manage a large company's change management initiative.", 
#     "I am worried about the performance of my SQL database.", 
#     "I am a Java developer who wants to pivot to Python development.", 
#     "How do I manage a team of salespeople?", 
#     "I need to be a better negotiator with strategic partnerships", 
#     "How do I market my products on Facebook and Instagram?", 
#     "I have been tasked with overseeing the implementation of a new ERP system for a multinational corporation.", 
#     "What are the best practices for leading a remote team of customer service representatives?", 
#     "I am responsible for facilitating the digital transformation initiative at a mid-sized enterprise.", 
#     "How can I improve the performance and management of my marketing team?"
#     "Help me get started with infrastructure automation."
# ]


# @pytest.fixture
# def setup():
#     pass

# @pytest.mark.run_every_commit