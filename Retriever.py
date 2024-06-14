"""
This module contains the Retriever class for the Chain framework.
This class is responsible for retrieving data from:
- a document or set of documents (i.e. a document loader)
- a CSV file
- a database
- a vector store

A Retriever object is not exactly a Tool, since it can be used in setting up a prompt, or for other purposes.
But it can be invoked like a Tool by an agent.
"""

class Retriever(): # TBD
    """
    Retriever class for the Chain framework.
    """
    def __init__(self):
        pass

    def __repr__(self):
        return Chain.standard_repr(self)

class DocumentLoader(Retriever): # TBD
    """
    DocumentRetriever class for the Chain framework.
    """
    def __init__(self):
        pass

class CSVLoader(Retriever):
    """
    CSVLoader class for the Chain framework.
    """
    def __init__(self):
        pass

class DatabaseLoader(Retriever):
    """
    DatabaseLoader class for the Chain framework.
    """
    def __init__(self):
        pass

