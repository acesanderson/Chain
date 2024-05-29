"""
This leverages several persona-defined prompts (publisher, bootcamp, L&D).
I'm attempting to get well-formed json. Will be using this to test llm models for how well they can generate structured data.
"""
from Chain import Chain, Model, Prompt, Parser

curation_prompt_publisher = Prompt("""
The company you work for publishes video courses, and has a library of thousands of courses for a variety of topics ranging from business, creative, and IT/software dev. You are being asked to create a learning path of existing courses that teach someone a fundamental skill.

Please create a curriculum for a training program on this topic: {{topic}}.

Map out 4-6 modules that comprehensively cover the most important topics for a working professional. Describe the curriculum generally; don"t go into detail about the instructional approach; and assume that these modules will be video courses. For each module, you will define the topic (as a short noun phrase, like "user experience design"), your rationale for picking the topic (1-3 sentences), and a description of the topic (1-3 sentences)
""")

curation_prompt_bootcamp = Prompt("""
You work for an educational bootcamp company that upskills early-career people in fundamental skills, ranging from business (front line worker to executive leadership, marketing/finance, etc.), creative (graphic design, video editing), and technology (software dev / IT administration) topics. You are being asked to come up with a learning path of video courses.

Please define a learning path on the topic of {{topic}} that will set beginners up for a career in the relevant field.

Map out 4-6 modules that comprehensively cover the most important topics for a working professional. Describe the curriculum generally; don"t go into detail about the instructional approach; and assume that these modules will be video courses. For each module, you will define the topic (as a short noun phrase, like "user experience design"), your rationale for picking the topic (1-3 sentences), and a description of the topic (1-3 sentences)
""")

curation_prompt_lnd = Prompt("""
You work as the learning and development administrator for a large enterprise company (over 1,000 employees), and you have been asked by leadership to develop a learning path of video courses that will address the most important skills for your company.

Please define a learning path on the topic of {{topic}} that will upskill your employees to be more effective.

Map out 4-6 modules that comprehensively cover the most important topics for a working professional. Describe the curriculum generally; don"t go into detail about the instructional approach; and assume that these modules will be video courses. For each module, you will define the topic (as a short noun phrase, like "user experience design"), your rationale for picking the topic (1-3 sentences), and a description of the topic (1-3 sentences)
""")

curation_prompt_transcript = Prompt("""
You work as the learning and development administrator for a large enterprise company (over 1,000 employees), and you have been asked by leadership to develop a learning path of video courses that will address the most important skills for your company.

Below is a transcript for a course on the topic of {{topic}}. Please analyze the transcript and then imagine this is the first course in a series of 4-6 courses. Think about how this intro course might be setting the scene for the courses to come. What topics would those courses be on?

TRANSCRIPT
==========
{{transcript}}
==========
""")

if __name__ == '__main__':
	topic = "Python for Machine Learning"
	parser = Parser(parser='curriculum_parser')
	model = Model('mistral')
	
	curation_publisher_chain = Chain(curation_prompt_publisher, model, parser)
	curation_bootcamp_chain = Chain(curation_prompt_bootcamp, model, parser)
	curation_lnd_chain = Chain(curation_prompt_lnd, model, parser)

	publisher = curation_publisher_chain.run(topic)
	bootcamp = curation_bootcamp_chain.run(topic)
	lnd = curation_lnd_chain.run(topic)

	curations = []
	curations.append(publisher)
	curations.append(bootcamp)
	curations.append(lnd)
