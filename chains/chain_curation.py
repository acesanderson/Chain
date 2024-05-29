"""
"""
from Chain import Chain, Model, Parser, Prompt
import random # to pick a random skill for demonstration purposes

topic_prompt = """
    Come up with a 5-10 module curriculum on the topic of {{topic}}.
    """

critique_prompt = """
        You are an L&D admin who has been asked to critique a proposed curriculum on the topic of {{topic}}.
        Here is the curriculum:
        ======================
        {{curriculum}}
        ======================
        Your answer should be entirely about the sequence of topics presented in the curriculum, and whether topics should be added or removed.
        """

recommendation_prompt = """
    A university is putting together a training curriculum.
    They have received three critiques of the curriculum. Here is the original curriculum:
    =====
    {{curriculum}}
    =====                    
    And here are the critiques:
    {% for critique in critiques %}
    =====
    Critique {{ loop.index }}:\n{{ critique }}
    =====
    {% endfor %}
    Review the above critiques, and using your best judgment, summarize the critiques and provide\n
    a recommendation to the university on how to proceed with the curriculum. Note: you can ignore some\n
    details of the critiques if they conflict or if you disagree with the particular recommendations.\n
    """

make_edits_prompt = """
    You are a curriculum developer. You have been given the following curriculum:
    ======================
    {{curriculum}}
    ======================

    You have been asked to make edits to the curriculum based on the following recommendation:
    ======================
    {{recommendation}}
    ======================
    Please rework the curriculum to incorporate the recommendation. You will want the curriculum to only have 5-10 overall modules, so you will
    have to be selective in what you include.
    Your answer will be only the revised curriculum.
    Include the following:
    - the original topic for the entire curriculum
    - the curriculum (module number, topic, your rational for including this topic, and a description for each module)
    """

# topic = "Python for Machine Learning"
# model = 'gpt-3.5-turbo-0125'
# parser = 'curriculum_parser'

def chain_curation(topic, critics = 2, model = 'gpt-3.5-turbo-0125'):
    """
    Takes a topic and returns a curated curriculum.
    Curriculum goes through a chain of critics, then is reconsolidated based on their feedback.
    """
    # Initialize the chains
    topic_chain = Chain(Prompt(topic_prompt),Model(model))
    critique_chain = Chain(Prompt(critique_prompt),Model(model))
    recommendation_chain = Chain(Prompt(recommendation_prompt),Model(model))
    # we want our final chain to output json. We can use the curriculum_parser for this.
    make_edits_chain = Chain(Prompt(make_edits_prompt),Model(model),Parser('curriculum_parser'))
    # Run the chains
    print(f'Generating curriculum for {topic}...')
    curriculum = topic_chain.run({'topic': topic})
    critiques = []
    for i in range(critics):
        print(f'Submitting curriculum for critique: reviewer #{i+1}...')
        critiques.append(critique_chain.run({'topic': topic,'curriculum': curriculum}))
    print('Generating recommendation...')
    recommendation = recommendation_chain.run({'curriculum': curriculum, 'critiques': critiques})
    print('Revising curriculum...')
    edited_curriculum = make_edits_chain.run({'curriculum': curriculum, 'recommendation': recommendation})
    return edited_curriculum

if __name__ == '__main__':
    topic = random.choice([                     # Randomly select a topic for demonstration purposes
        "Accounting",
        "User Experience Design",
        "Data Analysis",
        "Project Management",
        "Software Development",
        "Digital Marketing",
        "Cybersecurity",
        "Sales",
        "Customer Service",
        "Human Resources Management",
        "Graphic Design",
        "Public Speaking",
        "Business Strategy",
        "Machine Learning",
        "Supply Chain Management",
        "SEO/SEM",
        "Legal Compliance",
        "Environmental Management",
        "Biotechnology"
    ])
    result = chain_curation(topic,critics=2)    # Run the chain; just two critics because this is a demo.
    print(result)
