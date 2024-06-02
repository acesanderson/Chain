"""
This is a wrapper for the various data sources I want to leverage as part of my curation and content generation work.
So far we have two datasets:
- Course Descriptions (in a vector database). This has arbitrary IDs and the embeddings are course titles with descriptions.
- Course Transcripts (in a vector database). This is only about half of the transcripts that I was able to download. 300MB+

Notes:
- Should refactor descriptions so that ids are course titles.
- Should add a metadata database so that we can get richer results. (mongodb)

Usage:

`from Course_Data import query_descriptions, query_transcripts`
"""
import chromadb

descriptions_collection_name = "All_Courses-5-23-2024"
descriptions_persist_directory = '/home/bianders/Brian_Code/Chain_Framework/data/vectordbs/Library_RAG_Chain'
descriptions_client = chromadb.PersistentClient(path=descriptions_persist_directory)
descriptions_collection = descriptions_client.get_collection(name=descriptions_collection_name)

def query_descriptions(query, n_results=10):
    q = descriptions_collection.query(
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
