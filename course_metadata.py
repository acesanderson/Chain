"""
This script reads the cosmo export, loads into a dataframe, initializes as Course objects, and then passes to MongoDB.
This data can now be used by Course_Data.py.
"""

import pandas as pd
from dataclasses import dataclass
from pymongo import MongoClient

client = MongoClient("localhost", 27017)
# creeate a database
db=client.courses
# create a collection
courses_db = db.courses

# syntax examples for pymongo
# courses_db.insert_one({'name': 'mike', 'age': 30})
# print([p for p in courses.find() if 'name' == 'mike'])
# courses_db.find({'name': 'mike'})
# db.my_collection.find_one()
# courses.delete_many({})
# result = courses_db.insert_many()

excel_file = 'data/exports/courselist_en_US.xlsx'
df = pd.read_excel(excel_file)
# Convert all columns to string type before applying fillna
df = df.astype(str).fillna('')

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

# Filter to get active courses with release date after 2018-01-01 and locale en_US
active_courses = df[(df['Activation Status'] == 'ACTIVE') &
                    (df['Course Release Date'] > '2018-01-01') &
                    (df['Locale'] == 'en_US')]

# create a quick dict that maps the var names from our Course dataclass to the actual rows in our excel df.
variables = {}
columns = df.columns
for column in columns:
    variables[column.replace(' ', '_')] = column

# create a list of Course objects
course_variables = Course.__match_args__        # This is the variables we defined in our dataclass above.
courses = []
for row in active_courses.iterrows():
    values = {}
    for var in course_variables:
        values[var] = row[1][variables[var]]
    courses.append(Course(**values))

result = courses_db.insert_many([c.__dict__ for c in courses])
