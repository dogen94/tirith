import sqlite3


def connect_db():
    # Connect to SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect('db/oracle-card.db')
    # Create a cursor object using cursor() method
    cursor = conn.cursor()
    # Create a table
    cursor.execute('''CREATE TABLE IF NOT EXISTS cards
                    (id CHAR(36) PRIMARY KEY, name TEXT, age INTEGER)''')
    return conn, cursor


def insert_db(conn, cursor):
    # Insert data into table
    cursor.execute("INSERT INTO users (name, age) VALUES (?, ?)", ('Alice', 30))
    # Commit changes
    conn.commit()


def close_db(conn, cursor):
    # Close the cursor and connection
    cursor.close()
    conn.close()

