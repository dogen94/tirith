import sqlite3
import numpy as np

from .util import read_json, convert_legals, get_usd

DBDIR = 'db/'
JSONCOLS = [
    "oracle_id",
    "name",
    "mana_cost",
    "cmc",
    "type_line",
    "colors",
    "power",
    "toughness",
    "keywords",
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

    def write_preprocessed_db(self, dbout, **kw):
        self.preprocess_data(dbout, **kw)


    def preprocess_data(self, dbout, table, cols=JSONCOLS):
        for col in cols:
            # Get col from self
            exec_str = "SELECT " + "%s "% col + "FROM cards"
            self.cursor.execute(exec_str)
            data = self.cursor.fetchall()
            # Build numpy array out of data
            data_arr = np.array(data)
            # Preprocess
            preprocess = PREPROCESS.get(col, None)
            if preprocess:
                func = preprocess["func"]
                kws = preprocess.get("kw", {})
                pp_data = func(data_arr, **kws)
            else:
                pp_data = data_arr
            # Need to insert row by row.
            # Insert data to dbout
            dbout.insert2table(table, [col], pp_data)
        dbout.close_db()


def preprocess_string(data, remove=[","], replace=None):
    if replace:
        for exist, new in replace.items():
            if exist == None:
                if data == None:
                    data = new
            else:
                data = data.replace(exist, new)
                
    if remove:
        for c in remove:
            try:
                data = data.replace(c, "")
            except:
                breakpoint()
    return data

def preprocess_int(data, replace=None):
    if replace:
        for exist, new in replace.items():
            if (data == exist):
                    data = new
    return data


# Map column indices to their respective preprocessing functions
PREPROCESS = {
    "name": {"func": preprocess_string,
             "kw": {"remove": ",",
                    "replace": {None: None}}
            },
    "mana_cost": {"func": preprocess_string,
             "kw": {"remove": ",",
                    "replace": {None: "0"}},
            },
    "colors": {"func": preprocess_string,
               "kw": {"remove": ",",
                      "replace": {None: "0"}
                     }
            },
    "power": {"func": preprocess_int,
              "kw": {"replace": {None: -2,
                                 "*": -1,
                                 "*+1": -1
                                 }
                    }
            },
    "toughness": {"func": preprocess_int,
                  "kw": {"replace": {None: -2,
                                    "*": -1,
                                    "*+1": -1
                                    }
                        }
            },
    "prices": {"func": preprocess_int,
               "kw": {"replace": {None: np.NaN}
                     }
            },
}


def preprocess_data(raw_db, processed_db):
    try:
        # Get column names from raw_table
        raw_db.cursor.execute("PRAGMA table_info(cards)")
        columns = [row[1] for row in raw_db.cursor.fetchall()]

        # Create a dictionary for column names to column indices
        column_indices = {name: idx for idx, name in enumerate(columns)}

        # Read raw data
        raw_db.cursor.execute("SELECT * FROM cards")
        raw_data = raw_db.cursor.fetchall()

        # Preprocess data
        preprocessed_data = []
        for record in raw_data:
            processed_record = []
            for column_name, value in zip(columns, record):
                # Apply the preprocessing function based on column name
                preprocesses = PREPROCESS.get(column_name, False)
                if preprocesses:
                    ppkw = preprocesses.get("kw", {})
                    preprocess_func = preprocesses.get("func")
                    processed_value = preprocess_func(value, **ppkw)
                else:
                    processed_value = value
                processed_record.append(processed_value)
            preprocessed_data.append(tuple(processed_record))

        # Insert preprocessed data into the preprocessed database
        placeholders = ', '.join('?' * len(preprocessed_data[0]))
        insert_query = f"INSERT INTO cards ({', '.join(columns)}) VALUES ({placeholders})"
        processed_db.cursor.executemany(insert_query, preprocessed_data)
        
        # Commit changes
        processed_db.conn.commit()

    except Exception as e:
        print(f"An error occurred: {e}")
        processed_db.conn.rollback()  # Rollback in case of an error

    finally:
        # Close connections
        raw_db.close_db()
        processed_db.close_db()


def fill_table(db, fjson, table, **kw):
    cols = kw.pop("cols", db.defns.keys())
    jout = read_json(fjson)
    addcols = kw.pop("addcols", {})
    for entry in jout:
        vals = []
        ocols = []
        for col in cols:
            v = entry.get(col, None)
            if v == [] or v == "":
                v = None;
            if isinstance(v, list):
                v = ",".join(v)
            # Run function hook
            if db.defns[col].get("func"):
                v = db.defns[col].get("func")(v)
            # Run column key translation
            ocol = db.defns[col].get("trans", col)
            vals.extend([v])
            ocols.extend([ocol])
        # Check for additional cols
        if addcols:
            for addcol,addv in addcols:
                vals.extend([addv])
                ocols.extend([addcol])
        db.insert2table(table, ocols, vals)


