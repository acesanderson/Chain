from pymongo import MongoClient

client = MongoClient("localhost", 27017)

# creeate a database
db=client.courses
# create a collection
courses = db.courses
# add a record to our collection
courses.insert_one({'name': 'mike', 'age': 30})

print([p for p in courses.find() if 'name' == 'mike'])

courses.find({'name': 'mike'})