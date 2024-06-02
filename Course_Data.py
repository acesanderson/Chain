"""
This is a wrapper for the various data sources I want to leverage as part of my curation and content generation work.
So far we have two datasets:
- Course Descriptions (in a vector database). This has arbitrary IDs and the embeddings are course titles with descriptions.
- Course Transcripts (in a vector database). This is only about half of the transcripts that I was able to download. 300MB+
- courses_db: a MongoDB database with all of our Course objects.

Notes:
- Should refactor descriptions so that ids are course titles.

Usage:

`from Course_Data import query_descriptions, query_transcripts, get_mongodb_client, load_courses, Course`
`courses_db = get_mongodb_client()`
`courses = load_courses()`
"""
import chromadb
from dataclasses import dataclass
from pymongo import MongoClient

# Initialize our Course class.
@dataclass
class Course:
    Course_ID : str
    Course_Name : str
    Course_Name_EN : str
    Activation_Status : str
    Course_Description : str
    Course_Short_Description : str
    Instructor_Name : str
    Instructor_Short_Bio : str
    Course_Release_Date : str
    Course_Updated_Date : str
    LIL_URL : str
    LI_Level_EN : str
    Internal_Library : str
    Internal_Subject : str
    Primary_Software : str
    Visible_Duration : float
    Visible_Video_Count : str
    Contract_Type : str

# Initialize our vector databases (descriptions and transcripts)
short_descriptions_collection_name = "Short_Descriptions_5_23_2024"
short_descriptions_persist_directory = '/home/bianders/Brian_Code/Chain_Framework/data/vectordbs/Library_RAG_Chain'
short_descriptions_client = chromadb.PersistentClient(path=short_descriptions_persist_directory)
short_descriptions_collection = short_descriptions_client.get_collection(name=short_descriptions_collection_name)

def query_short_descriptions(query, n_results=10):
    """
    This currently returns the first document in the query results.
    You can imagine some use cases where you want the ids.
    In future, that can be iomplemented if necessary by changing 'documents' to 'ids' in the return.
    """
    q = short_descriptions_collection.query(
        query_texts=[query],
        n_results=n_results,
    )
    return q['documents'][0]

transcripts_collection_name = "Half_of_Transcripts_6-1-2024"
transcripts_persist_directory = '/home/bianders/Brian_Code/Chain_Framework/data/vectordbs/Transcripts'
transcripts_client = chromadb.PersistentClient(path=transcripts_persist_directory)
transcripts_collection = transcripts_client.get_collection(name=transcripts_collection_name)

def query_transcripts(query, n_results=10):
    q = transcripts_collection.query(
        query_texts=[query],
        n_results=n_results,
    )
    return q['ids'][0]


# syntax examples for pymongo
# courses_db.insert_one({'name': 'mike', 'age': 30})
# print([p for p in courses.find() if 'name' == 'mike'])
# courses_db.find({'name': 'mike'})
# db.my_collection.find_one()
# courses.delete_many({})
# result = courses_db.insert_many()

# load all the records in courses_db and instantiate them as Course objects. Be sure to suppress the _id field, as that is not part of our Course class.
def get_mongodb_client():
    client = MongoClient("localhost", 27017)
    db=client.courses
    courses_db = db.courses
    return courses_db

def load_courses():
    client = MongoClient("localhost", 27017)
    db=client.courses
    courses_db = db.courses    
    courses = []
    for course in courses_db.find({}, {'_id': 0}):
        courses.append(Course(**course))
    return courses

