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

class Database(object):
    def __init__(self, dbdir):
        self.dbdir = dbdir
        conn, cursor = self.connect_db(dbdir)
        self.conn = conn
        self.cursor = cursor
        self.tables = []
        self.set_table()
        self.defns = {}
        self.primary = []


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
        tabledefn = kw.pop("defns", self.defns)
        defn_list = []
        prim_list = []
        # Go through provided cols
        for i,(k0,d) in enumerate(tabledefn.items()):
            v = d["dtype"]
            k = self.defns[k0].get("trans", k0)
            defn_list.extend([f"{k} {v}"])
            if self.defns[k0].get("primary", False):
                prim_list.extend([f"{k}"])
            elif i == 0:
                self.set_primary(k)
                prim_list.extend([f"{k}"])
        # Assemble col string
        defn_str = " ,".join(defn_list)
        prim_str = "PRIMARY KEY (" + ", ".join(prim_list)  + ")"
        exec_str += defn_str + ", "+ prim_str + ")"
        return exec_str

    def genr8_insert_exec(self, table, ocols):
        if (table not in self.tables):
            raise Exception(f"Table {table} does not exist")
        istr = f"INSERT INTO {table} ("
        istr += ", ".join(ocols) + ") "
        qs = ["?"] * len(ocols)
        istr += "VALUES (" + ", ".join(qs) + ")"
        return istr

    def insert2table(self, table, ocols, vals):
        input_str = self.genr8_insert_exec(table, ocols)
        # Insert data into table
        self.cursor.execute(input_str, vals)
        # Commit changes
        self.conn.commit()

    def set_defns(self, defns, force = False):
        if (not self.defns or (self.defns and force)):
            self.defns = defns
            for defn in defns:
                pcol = self.defns[defn].get("primary", False)
                if pcol:
                    self.primary.append(defn)

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
            self.primary.append(col)

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
            if db.defn[col].get("func"):
                v = db.defn[col].get("func")(v)
            ocol = db.defn[col].get("trans", col)
            vals.extend([v])
            ocols.extend([ocol])
        # if None in vals or None in ocols:
        #     continue
        # else:
        db.insert2table("cards", ocols, vals)
