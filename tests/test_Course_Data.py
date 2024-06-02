"""
Decorators:
@pytest.mark.run_every_commit
@pytest.mark.run_occasionally
"""
import pytest
from Course_Data import query_short_descriptions, query_long_descriptions, query_transcripts, get_mongodb_client, load_courses
import random

queries = [
    "I have been hired to manage a large company's change management initiative.", 
    "I am worried about the performance of my SQL database.", 
    "I am a Java developer who wants to pivot to Python development.", 
    "How do I manage a team of salespeople?", 
    "I need to be a better negotiator with strategic partnerships", 
    "How do I market my products on Facebook and Instagram?", 
    "I have been tasked with overseeing the implementation of a new ERP system for a multinational corporation.", 
    "I am concerned about the scalability and performance issues of my PostgreSQL database.", 
    "As a C# developer, I am interested in transitioning to full-stack development using JavaScript and Node.js.", 
    "What are the best practices for leading a remote team of customer service representatives?", 
    "I need to enhance my skills in closing high-stakes deals with key industry partners.", 
    "How do I effectively promote my brand on LinkedIn and Twitter?", 
    "I am responsible for facilitating the digital transformation initiative at a mid-sized enterprise.", 
    "I am facing challenges with the response time of my NoSQL database.", 
    "As a web developer, I want to gain expertise in mobile app development with React Native.", 
    "How can I improve the performance and management of my marketing team?"
    ]

@pytest.fixture
def setup():
    pass

@pytest.mark.run_every_commit
def test_query_short_descriptions(setup):
    # Test case 1: Query the database and check the returned documents
    query = random.choice(queries)
    n_results = 5
    documents = query_short_descriptions(query, n_results)
    assert len(documents) == n_results

@pytest.mark.run_every_commit
def test_query_long_descriptions(setup):
    # Test case 1: Query the database and check the returned documents
    query = random.choice(queries)
    n_results = 5
    documents = query_long_descriptions(query, n_results)
    assert len(documents) == n_results

@pytest.mark.run_every_commit
def test_query_transcripts(setup):
    # Test case 1: Query the database and check the returned documents
    query = random.choice(queries)
    n_results = 5
    documents = query_transcripts(query, n_results)
    assert len(documents) == n_results

@pytest.mark.run_every_commit
def test_get_mongodb_client(setup):
    # Test case 1: Check if the MongoDB client is connected
    client = get_mongodb_client()
    assert client.find_one() is not None
    assert client.find_one()['Course_Name_EN'] is not None

@pytest.mark.run_every_commit
def test_load_courses(setup):
    # Test case 1: Load the courses from the MongoDB database
    courses = load_courses()
    assert len(courses) > 0
    assert courses[0].Course_Name_EN is not None

