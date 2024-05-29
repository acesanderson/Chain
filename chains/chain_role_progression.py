from Chain import Chain, Model, Prompt, Parser

curation_prompt = """
You are an L&D admin at a large enterprise company.
You are tasked with identifying the most important skills for different areas of the organization.
For this topic: {{topic}}, what are the 4-6 most important skills to develop?
Your answer should be a numbered list of skills, with a colon followed by a 1-2 sentence description of why it's important.
"""

role_progression_prompt = """
**System Prompt:**

Develop a comprehensive role progression framework tailored to specific industries or subject areas, defining competencies and learning outcomes for different professional levels based on user-provided skills.

**User Input Section:**

Topic Area: {{topic}}

Core Skills: {{coreskills}}

**Automated Generation Sections:**

Develop a Concise Description

Craft a concise description of the topic area. This should summarize the expected soft and technical skills and outline the function and role of this skill within an organization.
  
Develop Competencies for Each Skill:

For each user-provided core skill, develop 2-4 technical competencies that detail the specific abilities and knowledge necessary for professionals working in the topic area.

Additionally, include one soft skill competency for each core skill, focusing on interpersonal and adaptive skills that complement the technical aspects.

Each competency should have a short title and a one sentence description.

Define Professional Roles:

Outline typical career progression by defining roles at various levels (e.g., Junior, Mid-Level, Senior, Executive) appropriate to the specified industry.  

Craft Example Learning Outcomes:

For the first competency in the framework develop specific, measurable learning outcomes. These outcomes should articulate how an employee at that level will apply the competencies in their daily job functions, demonstrating the evolution and application of these competencies as employees advance in their careers. Each learning outcome should be written in third person starting with a verb that ends in "s".
"""

if __name__ == '__main__':
    topic = "Cloud-based Machine Learning"
    curation_Chain = Chain(Prompt(curation_prompt), Model('mistral'))
    role_progression_Chain = Chain(Prompt(role_progression_prompt), Model('mistral'))
    result = role_progression_Chain.run({'topic':topic,'coreskills':curation_Chain.run({'topic':topic})})
    print(result)

