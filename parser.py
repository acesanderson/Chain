"""
Thanks to Instructor library, this is very simple.
Edge cases (models and use cases not covered in Instructor library) can be addressed in the individual client libraries.
"""

from pydantic import BaseModel

class Parser():
	"""
	Parser class for use with Instructor and Pydantic models.
	"""
	def __init__(self, pydantic_model: BaseModel):
		self.pydantic_model = pydantic_model

	def __repr__(self):
		return f"Parser(pydantic_model={self.pydantic_model.__name__})"