import sqlite3
import os


DBDIR = 'db/'
JSONCOLS = [
    "oracle_id",
    "name",
    "mana_cost",
    "cmc",
    "type_line",
    "oracle_text",
    "legalities",
    "games",
    "set_id",
    "prices",
]


def connect_db(dbdir):
    # Connect to SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect(dbdir)
    # Create a cursor object using cursor() method
    cursor = conn.cursor()
    return conn, cursor

def create_table(conn, cursor, **kw):
    exec_str = genr8_table_exec(**kw)
    # Create a table
    cursor.execute(exec_str)

def genr8_table_exec(**kw):
    # Get title
    title = kw.pop("title")
    # Start exec string
    exec_str = f"'''CREATE TABLE IF NOT EXISTS {title} ("
    cols = kw.pop("cols", {})
    col_list = []
    # Go through provided cols
    for i,(k,v) in enumerate(cols.items()):
        if i == 0:
            col_list += [f"{k} {v} PRIMARY KEY"]
        else:
            col_list += f"{k} {v}"
    # Assemble col string
    col_str = ",".join(col_list)
    exec_str += col_str + ")'''"
    return exec_str

def insert_db(conn, cursor):
    # Insert data into table
    cursor.execute("INSERT INTO users (name, age) VALUES (?, ?)", ('Alice', 30))
    # Commit changes
    conn.commit()

def close_db(conn, cursor):
    # Close the cursor and connection
    cursor.close()
    conn.close()

