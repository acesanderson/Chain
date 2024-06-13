"""
This is a reworking of my original Langchain library_RAG script.
This accesses a vector store of LiL courses.
STATUS (5-23-2024):
- The vector store is set up (Course_Descriptions module)
- The query function is working
- I have used my Prompt and Parsers classes effectively in adding context to the prompt.
- Results from vectorDB are surprisingly not as good as I thought they would be.
NEXT STEPS:
- link this with chain_structured_curation so that I can I run it iteratively to create a json output like this:
 - {'topic', 'curriculum': [{'module_number', 'topic', 'courses': ['course1', 'course2', 'course3']}, {'module_number', 'topic', 'courses': ['course1', 'course2', 'course3']}, {'module_number', 'topic', 'courses': ['course1', 'course2', 'course3']}, {'module_number', 'topic', 'courses': ['course1', 'course2', 'course3']}, {'module_number', 'topic', 'courses': ['course1', 'course2', 'course3']}, {'module_number', 'topic', 'courses': ['course1', 'course2', 'course3']}]}
 - have another llm pick the best of three there in the context of the entire curriculum.
"""
from Chain import Chain, Model, Prompt, Parser
from Course_Descriptions import query_db                # This is my vector database of courses, and the query function

# Starting with one json object from chain_structured_curation.py
# (where I asked the three personas to create a curriculum on Difficult Conversations for Managers)
json = {'topic': 'Python for Machine Learning', 'curriculum': [{'module_number': 1, 'topic': 'Introduction to Python Programming', 'rationale': 'A solid foundation in Python programming is essential for employees to effectively work with machine learning tools and libraries.', 'description': 'This module covers the basics of Python programming, including data types, control structures, functions, and object-oriented programming concepts.'}, {'module_number': 2, 'topic': 'Data Manipulation and Analysis with NumPy and Pandas', 'rationale': 'NumPy and Pandas are powerful libraries for working with data in Python, and are widely used in machine learning projects.', 'description': 'This module focuses on using NumPy and Pandas to manipulate, analyze, and visualize data, which are critical skills for preparing data for machine learning algorithms.'}, {'module_number': 3, 'topic': 'Machine Learning Fundamentals', 'rationale': 'Understanding the core concepts and terminology of machine learning is crucial for employees to effectively communicate and collaborate on machine learning projects.', 'description': 'This module introduces the fundamental concepts of machine learning, including supervised and unsupervised learning, model evaluation, and common algorithms such as linear regression and k-means clustering.'}, {'module_number': 4, 'topic': 'Supervised Learning with Scikit-learn', 'rationale': 'Scikit-learn is a popular machine learning library in Python, and proficiency in using it is valuable for employees working on machine learning projects.', 'description': 'This module covers supervised learning techniques using Scikit-learn, including decision trees, random forests, support vector machines, and neural networks.'}, {'module_number': 5, 'topic': 'Unsupervised Learning and Dimensionality Reduction', 'rationale': 'Unsupervised learning techniques are important for discovering patterns and structure in data, and dimensionality reduction is often used to preprocess data for machine learning algorithms.', 'description': 'This module explores unsupervised learning techniques such as clustering and principal component analysis, as well as dimensionality reduction methods like t-SNE and UMAP.'}]}
topic = json['topic']
module = json['curriculum'][2]['topic'] + ": " + json['curriculum'][2]['description']
toc = '\n'.join([x['topic'] for x in json['curriculum']])
prompt = """
You are an L&D admin at a large enterprise (1,000+ employees).
You have been tasked with curating courses to build a learning path on the topic of {{topic}}, which will will have this structure:

{{toc}}

Right now you are focusing on one of the modules mentioned above. Here is a description of that module:

{{module}}

Your assistant has helpfully come up with this list of 10 LinkedIn Learning courses that might serve this need;
however, you need to narrow it down to 3 courses that most directly serve the learning objective described above.
Here are those courses, along with a short description of each.

{{context}}

Please select the 3 courses that you believe best serve the learning objective described above.
Your answer should be a list of the course titles. Only return the list.
"""

# model = 'gpt-3.5-turbo-0125'
model = 'gpt'
parser = "list"

# now to build {{context}}
courses = query_db(module, n_results=10)
context = ""
for course in courses:
    context += "- " + course + "\n"

# Assemble our Chain!
chain = Chain(Prompt(prompt), Model(model), Parser(parser))

if __name__ == '__main__':
    resp = chain.run({'topic':topic, 'context':context, 'module':module, 'toc':toc})


