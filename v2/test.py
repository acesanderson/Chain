from model import Model
model = Model('mistral:latest')
print(model.query('name ten mammals'))