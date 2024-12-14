FROM python:3.12-slim

RUN mkdir /pytest_project/
COPY ./test-requirements.txt /pytest_project/
COPY ./requirements.txt /pytest_project/
COPY ./setup.py /pytest_project/

WORKDIR /pytest_project/

RUN pip install --upgrade pip
RUN pip install -e .
RUN pip3 install -r /pytest_project/test-requirements.txt
RUN pip3 install -r /pytest_project/requirements.txt


CMD ["pytest"]
ENV PYTHONDONTWRITEBYTECODE=true
