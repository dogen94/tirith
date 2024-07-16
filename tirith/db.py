import sqlite3
import os

from .util import read_json, convert_legals, get_usd

DBDIR = 'db/'
JSONCOLS = [
    "oracle_id",
    "name",
    "mana_cost",
    "cmc",
    "type_line",
    "oracle_text",
    "legalities",
    "set_id",
    "prices",
]

COLSINFO = {
    "oracle_id": {"dtype": "CHAR(36)"},
    "name": {"dtype": "TEXT"},
    "mana_cost": {"dtype": "INT"},
    "cmc": {"dtype": "INT"},
    "type_line": {"dtype": "TEXT"},
    "oracle_text": {"dtype": "TEXT"},
    "legalities": {"dtype": "CHAR(22)",
                   "func": convert_legals},
    "set_id": {"dtype": "CHAR(36)"},
    "prices": {"dtype": "FLOAT",
               "func": get_usd,
               "trans": "price_usd"},

}

# COLTYPE = {
#     "oracle_id": "CHAR(36)",
#     "name": "TEXT",
#     "mana_cost": "INT",
#     "cmc": "INT",
#     "type_line": "TEXT",
#     "oracle_text": "TEXT",
#     "legalities": "CHAR(22)",
#     "set_id": "CHAR(36)",
#     "prices": "FLOAT",
# }

# COLFUNCS = {
#     "legalities": convert_legals,
#     "prices": get_usd,
# }

# COLTRANS = {
#     "prices": "price_usd",
# }


class Database(object):
    def __init__(self, dbdir):
        self.dbdir = dbdir
        conn, cursor = self.connect_db(dbdir)
        self.conn = conn
        self.cursor = cursor
        self.tables = []
        self.set_table()
        self.info = None
        self.primary = None


    def connect_db(self, dbdir):
        # Connect to SQLite database (or create it if it doesn't exist)
        conn = sqlite3.connect(dbdir)
        # Create a cursor object using cursor() method
        cursor = conn.cursor()
        return conn, cursor

    def create_table(self, **kw):
        exec_str = self.genr8_table_exec(**kw)
        # Create a table
        self.cursor.execute(exec_str)

    def genr8_table_exec(self, **kw):
        # Get title
        title = kw.pop("title")
        if (title in self.tables):
            raise Exception(f"Table {title} already exists in db")
        else:
            self.tables.append(title)
        # Start exec string
        exec_str = f"CREATE TABLE IF NOT EXISTS {title} ("
        cols = kw.pop("cols", COLSINFO)
        col_list = []
        # Go through provided cols
        for i,(k,d) in enumerate(cols.items()):
            v = d["dtype"]
            k = self.info[k].get("trans", k)
            if i == 0:
                self.set_primary(k)
                col_list.extend([f"{k} {v} PRIMARY KEY"])
            else:
                col_list.extend([f"{k} {v}"])
        # Assemble col string
        col_str = " ,".join(col_list)
        exec_str += col_str + ")"
        return exec_str

    def genr8_insert_exec(self, table, ocols):
        if (table not in self.tables):
            raise Exception(f"Table {table} does not exist")
        istr = f"INSERT INTO {table} ("
        istr += " ,".join(ocols) + ") "
        qs = ["?"] * len(ocols)
        istr += "VALUES (" + " ,".join(qs) + ")"
        return istr

    def insert2table(self, table, ocols, vals):
        input_str = self.genr8_insert_exec(table, ocols)
        # Insert data into table
        self.cursor.execute(input_str, vals)
        # self.cursor.execute("INSERT INTO users (name, age) VALUES (?, ?)", ('Alice', 30))
        # Commit changes
        self.conn.commit()

    def set_info(self, info, force = False):
        if (not self.info or (self.info and force)):
            self.info = info

    def set_table(self):
        self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = self.cursor.fetchall()
        if tables:
            for table in tables:
                self.tables.extend(table)
        else:
            self.tables = []

    def set_primary(self, col):
        if (not self.primary):
            self.primary = col

    def close_db(self):
        # Close the cursor and connection
        self.cursor.close()
        self.conn.close()


def fill_table(db, fjson, **kw):
    cols = kw.pop("cols", JSONCOLS)
    jout = read_json(fjson)
    for entry in jout:
        vals = []
        ocols = []
        for col in cols:
            v = entry.get(col, None)
            if db.info[col].get("func"):
                v = db.info[col].get("func")(v)
            ocol = db.info[col].get("trans", col)
            vals.extend([v])
            ocols.extend([ocol])
        if None in vals or None in ocols:
            continue
        else:
            db.insert2table("cards", ocols, vals)
