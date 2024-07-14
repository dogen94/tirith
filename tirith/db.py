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
        self.table = None
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
        self.table = title
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

    def genr8_insert_exec(self,ocols, vals):
        istr = f"INSERT INTO {self.table} ("
        istr += " ,".join(ocols) + ") "
        qs = ["?"] * len(ocols)
        istr += "VALUES (" + " ,".join(qs) + ")"
        return istr

    def insert2table(self, ocols, vals):
        input_str = self.genr8_insert_exec(ocols, vals)
        # Insert data into table
        self.cursor.execute(input_str)
        # self.cursor.execute("INSERT INTO users (name, age) VALUES (?, ?)", ('Alice', 30))
        # Commit changes
        self.conn.commit()

    def set_info(self, info, force = False):
        if (not self.info or (self.info and force)):
            self.info = info

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
    for entry in jout[:2]:
        vals = []
        ocols = []
        for col in cols:
            v = entry[col]
            if db.info[col].get("func"):
                v = db.info[col].get("func")(v)
            ocol = db.info[col].get("trans", col)
            vals.extend([v])
            ocols.extend([ocol])
        db.insert2table(ocols, vals)
