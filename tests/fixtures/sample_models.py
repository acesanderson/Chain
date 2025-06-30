from pydantic import BaseModel

class MyTestPydanticContent(BaseModel):
    value: str
    number: int

class TestAnimal(BaseModel):
    name: str
    species: str
    age: int
    habitat: str

class TestFrog(BaseModel):
    species: str
    name: str
    legs: int
    color: str
