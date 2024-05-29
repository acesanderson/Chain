job_titles = """
## Leadership/Management
- Managers
- Directors
- Executives
- Leaders

## Engineering/Technical
- Engineers
- Software Developers/Engineers
- Systems Administrators
- Network Administrators
- Cybersecurity Analysts
- Database Administrators
- IT Support Specialists
- Computer Systems Analysts

## Analytics/Consulting
- Analysts
- Consultants
- Business Analysts
- Data Analysts
- Financial Analysts
- Management Consultants
- Market Research Analysts
- Operations Research Analysts
- Risk Management Analysts

## Sales/Customer Service
- Sales Representatives
- Customer Service Representatives
- Account Managers
- Client Relationship Managers
- Sales Operations Specialists
- Sales Enablement Specialists
- Sales Analytics Specialists
- Sales Training Specialists
- Channel Sales Specialists
- Solutions/Product Sales Specialists

## Marketing
- Marketing Specialists
- Digital Marketing Specialists
- Social Media Specialists
- Product Marketing Specialists
- Content Marketing Specialists
- Email Marketing Specialists
- Marketing Analytics Specialists
- Marketing Automation Specialists

## Finance/Accounting
- Accountants
- Auditors
- Budget Analysts
- Credit Analysts
- Financial Analysts

## Human Resources
- HR Specialists
- Compensation/Benefits Specialists
- Training & Development Specialists
- Recruiters

## Operations/Administration
- Administrative Assistants
- Office Clerks
- Data Entry Operators
- Warehouse/Logistics Workers
- Logistics Specialists
- Supply Chain Specialists

## Quality/Compliance
- Quality Assurance Specialists
- Compliance Specialists
- Regulatory Affairs Specialists

## Other Specialists
- Technical Writers
- Project Management Specialists
- Business Process Specialists
- Clinical Research Specialists (Healthcare/Pharma)
- User Experience (UX) Specialists
- Instructional Design Specialists
"""

### Job_titles multiline string is markdown formatted.
### Please create a dictionary that has the job categories (which are prepended with ##) as keys and the job titles as values.
### The job titles should be a list of strings.

### Expected output:
# {
#     'Leaders': ['Managers', 'Directors', 'Executives', 'Leaders'],
#     'Engineering/Technical': ['Engineers', 'Software Developers/Engineers', 'Systems Administrators', 'Network Administrators', 'Cybersecurity Analysts', 'Database Administrators', 'IT Support Specialists', 'Computer Systems Analysts'],
# }

job_titles_dict = {}
job_titles_lines = job_titles.split("\n")
job_category = ""
for line in job_titles_lines:
    if line.startswith("##"):
        job_category = line.replace("## ", "")
        job_titles_dict[job_category] = []
    elif line:
        job_titles_dict[job_category].append(line.replace("- ", ""))

from Chain import Chain, Model, Prompt, Parser

job_title_prompt = """
Please look at the data below and tell me what you think it is.

{{job_titles}}
"""

c=Chain(Prompt(job_title_prompt), Model('gpt-3.5-turbo'))

