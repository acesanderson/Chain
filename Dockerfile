FROM python:3.12-slim

RUN mkdir /pytest_project/
COPY ./test-requirements.txt /pytest_project/

RUN pip install --upgrade pip
RUN pip3 install -r /pytest_project/test-requirements.txt

WORKDIR /pytest_project/

CMD ["pytest"]
ENV PYTHONDONTWRITEBYTECODE=true
