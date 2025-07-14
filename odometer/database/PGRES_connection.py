"""
This module provides a context manager for connecting to a PostgreSQL database.
"""

import psycopg2
from contextlib import contextmanager
import subprocess
import os, sys

hostnames = {
    "remote": ["Botvinnik", "bianders-mn7180.linkedin.biz", "AlphaBlue"],
    "local": ["Caruana"],
}

# Constants
hostname = subprocess.check_output(["hostname"]).decode("utf-8").strip()
password = os.getenv("POSTGRES_PASSWORD")
if not password:
    print("POSTGRES_PASSWORD not found in environment variables.")
    sys.exit()


@contextmanager
def get_db_connection():
    """
    This is a context manager for the database connection.
    We create a new database connection for each operation, which is important for thread safety.
    """
    # print(hostname)
    if hostname in hostnames["remote"]:
        host = "10.0.0.82"
    elif hostname in hostnames["local"]:
        host = "localhost"
    else:  # Docker shenaningans await
        host = "10.0.0.82"
    connection = psycopg2.connect(
        dbname="kramer",
        host=host,
        port="5432",
        user="bianders",
        password=password,
    )
    try:
        yield connection
    finally:
        connection.close()
