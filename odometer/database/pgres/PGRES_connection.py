from contextlib import contextmanager
import os, sys, socket, subprocess, psycopg2

password = os.getenv("POSTGRES_PASSWORD")
if not password:
    print("POSTGRES_PASSWORD not found in environment variables.")
    sys.exit()


def get_postgres_host():
    endpoints = ["localhost", "10.0.0.82", "68.47.92.102"]

    for host in endpoints:
        try:
            socket.create_connection((host, 5432), timeout=1).close()
            return host
        except:
            continue

    raise ConnectionError("Cannot reach PostgreSQL server on any known endpoint")


@contextmanager
def get_db_connection():
    """
    This is a context manager for the database connection.
    We create a new database connection for each operation, which is important for thread safety.
    """
    host = get_postgres_host()
    connection = psycopg2.connect(
        dbname="chain",
        host=host,
        port="5432",
        user="bianders",
        password=password,
    )
    try:
        yield connection
    finally:
        connection.close()


# """
# This module provides a context manager for connecting to a PostgreSQL database.
# """
#
# import psycopg2
# from contextlib import contextmanager
# import subprocess
# import os
#
# hostnames = {
#     "remote": ["Botvinnik", "bianders-mn7180.linkedin.biz", "AlphaBlue"],
#     "local": ["Caruana"],
# }
#
# # Constants
# hostname = subprocess.check_output(["hostname"]).decode("utf-8").strip()
# password = os.getenv("POSTGRES_PASSWORD")
# if not password:
#     raise ValueError("POSTGRES_PASSWORD not found in environment variables.")
#
#
# @contextmanager
# def get_db_connection():
#     """
#     This is a context manager for the database connection.
#     We create a new database connection for each operation, which is important for thread safety.
#     """
#     # print(hostname)
#     if hostname in hostnames["remote"]:
#         host = "10.0.0.82"
#     elif hostname in hostnames["local"]:
#         host = "localhost"
#     else:  # Docker shenaningans await
#         host = "10.0.0.82"
#     connection = psycopg2.connect(
#         dbname="chain",
#         host=host,
#         port="5432",
#         user="bianders",
#         password=password,
#     )
#     try:
#         yield connection
#     finally:
#         connection.close()
