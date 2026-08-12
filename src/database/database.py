import sqlite3
from pathlib import Path


# ==================================================
# DATABASE CONFIGURATION
# ==================================================

DATABASE_PATH = Path(
    "data/database/insightai.db"
)


# ==================================================
# GET DATABASE CONNECTION
# ==================================================

def get_connection():
    """
    Create and return a connection to the
    InsightAI SQLite database.
    """

    # Make sure the database directory exists
    DATABASE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    return connection


# ==================================================
# EXECUTE SQL QUERY
# ==================================================

def execute_query(
    query: str,
):
    """
    Execute a read-only SQL query and return
    column names and rows.
    """

    connection = get_connection()

    try:

        cursor = connection.cursor()

        cursor.execute(query)

        rows = cursor.fetchall()

        # Get column names
        if cursor.description:

            columns = [
                description[0]
                for description in cursor.description
            ]

        else:

            columns = []

        return columns, rows

    finally:

        connection.close()