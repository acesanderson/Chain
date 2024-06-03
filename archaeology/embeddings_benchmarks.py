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
import re
import chromadb
import json         # for pretty printing dicts
import pandas as pd
from chromadb import Documents, EmbeddingFunction, Embeddings

## Load short course descriptions into memory
## This is from data/short_descriptions_db.py
excel_file = '/home/bianders/Brian_Code/Chain_Framework/data/exports/courselist_en_US.xlsx'
df = pd.read_excel(excel_file)

df = df.astype(str).fillna('')
active_courses = df[(df['Activation Status'] == 'ACTIVE') &
                    (df['Course Release Date'] > '2018-01-01') &
                    (df['Locale'] == 'en_US')]

# get only these two columns: "Course Name EN" and "Course Short Description"
# handle utf encoding
def clean_text(text):
    replacements = {
        'â€™': "'",
        'â€œ': '"',
        'â€˜': "'",
        'â€”': '—',  # Em-dash
        'â€“': '–'   # En-dash
    }
    for wrong, right in replacements.items():
        text = text.replace(wrong, right)
    return text

# clean the text, save as tuples (name, description)
# creating short descriptions list
short_descriptions = []
for index, row in active_courses.iterrows():
    clean_name = clean_text(row['Course Name EN'])
    clean_description = clean_text(row['Course Short Description'])
    short_descriptions.append((clean_name, clean_description))

# now creating long_descriptions list
long_descriptions = []
for index, row in active_courses.iterrows():
    clean_name = clean_text(row['Course Name EN'])
    clean_description = clean_text(row['Course Description'])
    long_descriptions.append((clean_name, clean_description))

# default model in chroma, which we've already used, is all-minilm
ollama_models = """mxbai-embed-large
nomic-embed-text
avr/sfr-embedding-mistral
hellord/e5-mistral-7b-instruct:Q4_0
snowflake-arctic-embed
snowflake-arctic-embed:22m
all-minilm:latest""".split('\n')

# create the database we will be loading our various embeddings into
embeddings_test_directory = '/home/bianders/Brian_Code/Chain_Framework/data/vectordbs/Embeddings_Test2'
embeddings_test_client = chromadb.PersistentClient(path=embeddings_test_directory)

short_descriptions_collection_template = "Short_Descriptions_5_23_2024_"
long_descriptions_collection_template = "Long_Descriptions_5_23_2024_"

# our functions

def get_embeddings(model, text):
    """
    Basic ollama embeddings function.
    We can specify model.
    """
    response = ollama.embeddings(model=model, prompt=text)
    return response['embedding']

def create_short_descriptions_collection_for_model(short_descriptions, model):
    """
    This creates a short descriptions VDB with a given ollama embeddings model.
    """
    if model == 'hellord/e5-mistral-7b-instruct:Q4_0':
        short_descriptions_collection_name = short_descriptions_collection_template + "e5-mistral-7bn-instruct"   # this model name breaks it for some reason!
    else:
        short_descriptions_collection_name = short_descriptions_collection_template + model.replace("/", "-").replace(":", "-").replace("_", "-") # collections can't have these special chars
    embeddings_test_client.create_collection(name=short_descriptions_collection_name)
    short_descriptions_collection = embeddings_test_client.get_collection(name=short_descriptions_collection_name)
    start = time()
    for index, short_description in enumerate(short_descriptions):
        document = short_description[0] + "::" + short_description[1]
        short_descriptions_collection.add(
            documents=[document],
            ids=[short_description[0]],
            embeddings = [get_embeddings(model, document)]
        )
        print(f"Added document {index + 1} of {len(short_descriptions)} to the database for model {model}.")
    end = time()
    duration = end - start
    return (short_descriptions_collection_name, duration)

def create_long_descriptions_collection_for_model(long_descriptions, model):
    """
    This creates a long descriptions VDB with a given ollama embeddings model.
    """
    if model == 'hellord/e5-mistral-7b-instruct:Q4_0':
        long_descriptions_collection_name = long_descriptions_collection_template + "e5-mistral-7bn-instruct"    # this model name breaks it for some reason!
    else:
        long_descriptions_collection_name = long_descriptions_collection_template + model.replace("/", "-").replace(":", "-").replace("_", "-") # collections can't have these special chars
    embeddings_test_client.create_collection(name=long_descriptions_collection_name)
    long_descriptions_collection = embeddings_test_client.get_collection(name=long_descriptions_collection_name)
    start = time()
    for index, long_description in enumerate(long_descriptions):
        document = long_description[0] + "::" + long_description[1]
        long_descriptions_collection.add(
            documents=[document],
            ids=[long_description[0]],
            embeddings = [get_embeddings(model, document)]
        )
        print(f"Added document {index + 1} of {len(long_descriptions)} to the database for model {model}.")
    end = time()
    duration = end - start
    return (long_descriptions_collection_name, duration)

# our function for querying the stuff
def query_descriptions(query, n_results=10, collection = "", model = ""):
    """
    This currently returns the first document in the query results.
    You can imagine some use cases where you want the ids.
    In future, that can be iomplemented if necessary by changing 'documents' to 'ids' in the return.
    """
    collection = embeddings_test_client.get_collection(name=collection)
    q = collection.query(
        query_embeddings=[get_embeddings(model,query)],
        n_results=n_results,
    )
    return q['documents'][0]

# test query function
# r1 = query_descriptions('I want to learn python.', n_results=10, collection = 'Short_Descriptions_5_23_2024_mxbai-embed-large', model='mxbai-embed-large')
# r2 = query_descriptions('I want to learn python.', n_results=10, collection = 'Short_Descriptions_5_23_2024_nomic-embed-text', model='nomic-embed-text')
# r3 = query_descriptions('I want to learn python.', n_results=10, collection = 'Short_Descriptions_5_23_2024_snowflake-arctic-embed', model='snowflake-arctic-embed')
# r4 = query_descriptions('I want to learn python.', n_results=10, collection = 'Short_Descriptions_5_23_2024_avr-sfr-embedding-mistral', model='avr/sfr-embedding-mistral')
# r5 = query_descriptions('I want to learn python.', n_results=10, collection = 'Short_Descriptions_5_23_2024_snowflake-arctic-embed-22m', model='snowflake-arctic-embed:22m')
# r6 = query_descriptions('I want to learn python.', n_results=10, collection = 'Short_Descriptions_5_23_2024_all-minilm-latest', model='all-minilm:latest')
# r7 = query_descriptions('I want to learn python.', n_results=10, collection = 'Short_Descriptions_5_23_2024_e5-mistral-7bn-instruct', model='hellord/e5-mistral-7b-instruct:Q4_0')

# if __name__ == "__main__":
#     results = {}
#     for model in ollama_models:
#         try:
#             collection = create_long_descriptions_collection_for_model(long_descriptions, model)  # It'll throw an error if collection already exists.
#             print(f"Model: {model} took {collection[1]} seconds to load the long descriptions.")
#             model_results = {}
#             model_results['collection'] = collection[0]
#             model_results['duration'] = collection[1]
#             results[model] = model_results
#         except:
#             print(f"Collection already exists for model {model}.")

queries = """"I want to learn Python programming."
"I want to start a career in Sales."
"I'm creating a startup and need to learn basic digital marketing techniquesl."
"I have been hired to manage a large company's change management initiative.", 
"I am worried about the performance of my SQL database.", 
"I am a Java developer who wants to pivot to Python development.", 
"How do I manage a team of salespeople?", 
"I need to be a better negotiator with strategic partnerships", 
"How do I market my products on Facebook and Instagram?", 
"I have been tasked with overseeing the implementation of a new ERP system for a multinational corporation.", 
"What are the best practices for leading a remote team of customer service representatives?", 
"I am responsible for facilitating the digital transformation initiative at a mid-sized enterprise.", 
"How can I improve the performance and management of my marketing team?"
Help me get started with infrastructure automation." """.split('\n')

#==============================================================================
"""
We will want the following:

dict of model and the corresponding collection names (short desc, long desc)
"""

collection_names = [c.name for c in embeddings_test_client.list_collections()]
collection_names.sort()

# split collection names into two lists, one that starts with Short_Descriptions and one that starts with Long_Descriptions
short_descriptions_collections = [c for c in collection_names if c.startswith('Short_Descriptions')]
long_descriptions_collections = [c for c in collection_names if c.startswith('Long_Descriptions')]
# alphabetize the collections
short_descriptions_collections.sort()
long_descriptions_collections.sort()
collections = list(zip(short_descriptions_collections, long_descriptions_collections))
ollama_models.sort()

models_iterator = list(zip(ollama_models, collections))

# results = {}
# for i, query in enumerate(queries):
#     print(f"Running query {i + 1} of {len(queries)}")
#     query_results = {}
#     for index, m in enumerate(models_iterator):
#         print(f"\tRunning model {index + 1} of {len(models_iterator)}: {m[0]}.")
#         for collection in m[1]:
#             try:
#                 result = query_descriptions(query, n_results=10, collection=collection, model=m[0])
#                 print(f"\t\tModel {m[0]} and collection {collection} returned {result}")
#                 query_results[collection] = result
#             except:
#                 print(f"Error for model {m[0]} and collection {collection}.")
#     results[query] = query_results

"""
query
    model
        collection

"""
"""
results schema:

{
"I want to start a career in Sales." : {
        "Short_Descriptions_5_23_2024_mxbai-embed-large" : [course1, course2, course3, course4, course5, course6, course7, course8, course9, course10],
        "Short_Descriptions_5_23_2024_nomic-embed-text" : [course1, course2, course3, course4, course5, course6, course7, course8, course9, course10],
        "Long_Descriptions_5_23_2024_mxbai-embed-large" : [course1, course2, course3, course4, course5, course6, course7, course8, course9, course10],
        "Long_Descriptions_5_23_2024_nomic-embed-text" : [course1, course2, course3, course4, course5, course6, course7, course8, course9, course10],
        },
"I want to learn Python programming." : {
    "short_descriptions" : {
        "Short_Descriptions_5_23_2024_mxbai-embed-large" : [course1, course2, course3, course4, course5, course6, course7, course8, course9, course10],
        "Short_Descriptions_5_23_2024_nomic-embed-text" : [course1, course2, course3, course4, course5, course6, course7, course8, course9, course10],
        }.
    "long_descriptions" : {
        "Long_Descriptions_5_23_2024_mxbai-embed-large" : [course1, course2, course3, course4, course5, course6, course7, course8, course9, course10],
        "Long_Descriptions_5_23_2024_nomic-embed-text" : [course1, course2, course3, course4, course5, course6, course7, course8, course9, course10],
        }.
}

"""

# save results as a json file to root directory
# with open('results.json', 'w') as f:
#     json.dump(results, f, indent=4)

# r = list(results.keys())
# cc = list(results[r[0]].keys())

# big_txt_file = ""
# for rr in r:
#     big_txt_file += f"\n{rr}\n"
#     for c in cc:
#         big_txt_file += f"\t{c}\n"
#         for i, course in enumerate(results[rr][c]):
#             big_txt_file += f"\t\t{re.findall('^(.+)::', course)[0]}\n"

OG_vdb = chromadb.PersistentClient(path='/home/bianders/Brian_Code/Chain_Framework/data/vectordbs/Library_RAG_Chain')
OG_vdb_collection = 'Short_Descriptions_5_23_2024'
OG_vdb_collection = OG_vdb.get_collection(name=OG_vdb_collection)

OG_results = {}
for query in queries:
    q = OG_vdb_collection.query(
        query_texts = [query],
        n_results=10,
    )
    OG_results[query] = q['ids']

OG_txt = ""
for q in queries:
    OG_txt += f"\n{q}\n"
    for course in OG_results[q]:
        OG_txt += f"\t{course}\n"

