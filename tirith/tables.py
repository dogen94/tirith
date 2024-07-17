from .db import Database, fill_table
from .util import convert_legals, get_usd

ORACLE_TABLE_DEFNS = {
    "oracle_id": {"dtype": "CHAR(36)",
                  "primary": True},
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

PRICE_TABLE_DEFNS = {
    "oracle_id": {"dtype": "CHAR(36)",
                  "primary": True,},
    "set_id": {"dtype": "CHAR(36)"},
    "prices": {"dtype": "FLOAT",
               "func": get_usd,
               "trans": "price_usd"},
    "date": {"dtype": "DATE",
             "primary": True}
}


def update_price_table():
    table_name = "prices"
    db = Database("db/oracle-prices.db")
    if table_name not in db.tables:
        raise Exception(f"Table {}")
    db.set_defns(PRICE_TABLE_DEFNS)
    db.create_table(title="prices")

    
