from Chain import Chain, Model, Prompt, Parser, Chat

model = "claude"

examples = """
1. Business schools for management and leadership content:
    - Wharton
    - MIT Sloan
    - Harvard Business School
    - Stanford Graduate School of Business
    - INSEAD
    - London Business School
    - Kellogg School of Management
    - Columbia Business School
    - Booth School of Business (University of Chicago)
    - Haas School of Business (UC Berkeley)
2. Tech companies for cloud computing and AI/ML content:
    - Amazon Web Services
    - Google Cloud
    - Microsoft Azure
    - IBM Cloud
    - Oracle Cloud
    - Salesforce
    - NVIDIA
    - OpenAI
    - DeepMind
    - Databricks
3. Cybersecurity organizations:
    - ISC²
    - SANS Institute
    - CompTIA
    - EC-Council
    - ISACA
    - Offensive Security
    - Black Hat
    - DEF CON
    - CrowdStrike
    - Symantec
4. Design firms for design thinking and UX content:
    - IDEO
    - frog
    - Designit
    - Fjord (Accenture Interactive)
    - Continuum
    - Ziba Design
    - Fuseproject
    - Lunar Design
    - Smart Design
    - Ammunition Group
5. Marketing organizations for digital marketing content:
    - American Marketing Association
    - HubSpot
    - Content Marketing Institute
    - MarketingProfs
    - Digital Marketing Institute
    - Moz
    - Semrush
    - Hootsuite
    - Sprout Social
    - Mailchimp
6. Financial training providers for financial analysis content:
    - Wall Street Prep
    - Training The Street
    - Corporate Finance Institute
    - Financial Edge Training
    - Fitch Learning
    - Moody's Analytics
    - Bloomberg Market Concepts
    - Udemy
    - Coursera
    - edX
7. Project management associations for agile and project management content:
    - PMI
    - Scrum Alliance
    - Agile Alliance
    - IPMA (International Project Management Association)
    - PRINCE2 (Axelos)
    - APM (Association for Project Management)
    - Lean Enterprise Institute
    - Disciplined Agile (DA)
    - Scaled Agile Framework (SAFe)
8. Supply chain management organizations:
    - APICS
    - CSCMP (Council of Supply Chain Management Professionals)
    - ISM (Institute for Supply Management)
    - ASCM (Association for Supply Chain Management)
    - Demand Driven Institute
    - Supply Chain Canada
    - Chartered Institute of Procurement & Supply (CIPS)
    - Gartner Supply Chain
    - Supply Chain Management Review
9. Human resources associations for HR and talent management content:
    - SHRM
    - ATD (Association for Talent Development)
    - CIPD (Chartered Institute of Personnel and Development)
    - WorldatWork
    - HR Certification Institute (HRCI)
    - International Public Management Association for Human Resources (IPMA-HR)
    - Academy of Human Resource Development (AHRD)
    - National Human Resources Association (NHRA)
    - International Association for Human Resource Information Management (IHRIM)
""".strip()

skills = """
Leadership
Project Management
Management
Communication
Business Analysis
Diversity & Inclusion
Finance
SQL
Python (Programming Language)
Data Science
Software Development
Data Analysis
Decision-Making
Training and Development (HR)
Marketing
Databases
Organizational Leadership
Talent Management
Human Resources (HR)
Sales
Microsoft Office
Customer Service
Cloud Computing
Time Management
Web Development
Customer Relationship Management (CRM)
Microsoft Power BI
Accounting
Adobe Photoshop
Agile Project Management
Personal Branding
Digital Marketing
System Administration
Machine Learning
Social Media Marketing
Business Intelligence (BI)
Technical Support
Project Management Software
Coaching
DevOps
User Experience Design (UED)
Statistics
Data Modeling
Object-Oriented Programming (OOP)
Recruiting
Front-End Development
Corporate Finance
E-Learning
Back-End Web Development
Application Programming Interfaces (API)
Brand Design
Database Administration
Personal Finance
Search Engine Optimization (SEO)
Cybersecurity
Product Design
Design Thinking
Java
Sales Management
JavaScript
""".strip().split('\n')

job_titles = """
Data Analyst
Cyber Security Analyst
Cyber Security Specialist
Data Engineer
Data Scientist
Data Specialist
Database Administrator
Full Stack Engineer
Information Technology Specialist
Java Software Engineer
Javascript Developer
Network Engineer
Python Developer
System Administrator
Web Developer
Software Engineer
Account Executive
People Manager
Executive Assistant
Chief of Staff
Human Resources Business Partner
Product Manager
Accountant
Business Analyst
Customer Service Representative
Business Development Associate
Customer Service Manager
HR Manager
Operations Manager
Social Media Manager
Supply Chain Specialist
Financial Analyst
Human Resources Specialist
Marketing Manager
Marketing Specialist
Program Manager
Project Manager
Recruiter
Sales Manager
Salesperson
User Experience Designer
Graphic Designer
Photographer
""".strip().split('\n')

prompt_string = """
My company is building a set of professional certificate programs for consumer and enterprise learners, across the most important skills.

Each certificate program will be a curated list of courses and resources from our existing course library, and is endorsed by the most respected organizations in the field.

Look at this list of skills and recommended organizations:

====================================
{{examples}}
====================================

For this skill: {{skill}}, please recommend the top ten organizations that we should partner with for each skill area.
These organizations should be well-known in the industry, have a strong reputation for quality thought leadership, and represent an aspirational place for learners to work.
"""

job_title_prompt_string = """
My company is building a set of professional certificate programs for consumer and enterprise learners, across the most important skills.

Each certificate program will be a curated list of courses and resources from our existing course library, and is endorsed by the most respected organizations in the field.

Look at this list of skills and recommended organizations:

====================================
{{examples}}
====================================

For this job title: {{job_title}}, please recommend the top ten organizations that we should partner with for each skill area.
These organizations should be well-known within the job title's professional community, have a strong reputation for quality thought leadership, and represent an aspirational place for learners to work.

Please provide ONLY a numbered list of the organization names, with no descriptions or explanations.
""".strip()

# chain = Chain(Prompt(prompt_string),Model(model))

# output = ""
# for index,skill in enumerate(skills):
#     print(f"Asking Claude about skill {index+1}: {skill}")
#     output += "\n====================================\n" + skill + "\n====================================\n\n"
#     r = chain.run({"skill": skill, "examples": examples})
#     output += r.content

# with open('output.txt', 'w') as f:
#     f.write(output) 

chain = Chain(Prompt(job_title_prompt_string),Model(model))

output2 = ""
for index, job_title in enumerate(job_titles):
    print(f"Asking Claude about job title {index+1} of {len(job_titles)}: {job_title}")
    output2 += "\n====================================\n" + job_title + "\n====================================\n\n"
    r = chain.run({"job_title": job_title, "examples": examples})
    output2 += r.content

