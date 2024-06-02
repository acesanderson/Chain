from glob import glob
import pandas as pd
from dataclasses import dataclass

excel_file = 'data/exports/courselist_en_US.xlsx'
df = pd.read_excel(excel_file)

@dataclass
class Course:
    Course_ID : str
    Course_Name : str
    Course_Name_EN : str
    Activation_Status : str
    Course_Description : str
    Course_Short_Description : str
    Instructor_Name : str
    Instructor_Short_Bio : str
    Course_Release_Date : str
    Course_Updated_Date : str
    LIL_URL : str
    LI_Level_EN : str
    Internal_Library : str
    Internal_Subject : str
    Primary_Software : str
    Visible_Duration : float
    Visible_Video_Count : str
    Contract_Type : str

column_names = Course.__match_args__

# grab all rows from df where Activation_Status is 'Active'
active_courses = df[df['Activation Status'] == 'ACTIVE']
# grab all rows from active_courses that have Course_Release_date later than 2018-01-01
active_courses = active_courses[active_courses['Course Release Date'] > '2018-01-01']
# grab all rows where Locale is 'en_US'
active_courses = active_courses[active_courses['Locale'] == 'en_US']

# create a quick dict that maps the var names from our Course dataclass to the actual rows in our excel df.
variables = {}
columns = df.columns
for column in columns:
    variables[column.replace(' ', '_')] = column

# create a list of Course objects
course_variables = Course.__match_args__
courses = []
for row in active_courses.iterrows():
    values = {}
    for var in course_variables:
        values[var] = row[1][variables[var]]
    courses.append(Course(**values)

