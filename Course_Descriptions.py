import chromadb

collection_name = "All_Courses-5-23-2024"
persist_directory = '/home/bianders/Brian_Code/llm_development/vectordbs/Library_RAG_Chain'
client = chromadb.PersistentClient(path=persist_directory)
collection = client.get_collection(name=collection_name)

def query_db(query, n_results=10):
    q = collection.query(
        query_texts=[query],
        n_results=n_results,
    )
    return q['documents'][0]

