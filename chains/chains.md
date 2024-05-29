### Chains

Here is where I document the types of chains I'm construction.

#### Chain Curation

Enter a topic and the following happens:
- curriculum is generated
- a number of "critics" is spun up to critique the curriculum
- a Researcher collects and reconciles the feedback
- an Editor implements the feedback, reduces the curriculum down, and returns it as a json object

#### Chain RAG

Uses the Course_Descriptions module (which loads query_db).
Enter a json curriculum and the following happens:
- an Assistant (the chroma database) comes up with a list of X courses
- the L&D admin (LLM) picks the courses that best fit the module description given to them.
- returns a list

#### Chain Role Progression

Enter a topic, and the following happens:
- a chain generates the core skills associated with the topic
- a chain takes that + the topic and generates a role progression framework

#### Chain Structured Curation

Enter a topic, and the following happens:
- an L&D admin, a bootcamp instructor, and a video publisher generate curricula as json objects.
