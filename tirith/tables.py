import datetime

from .db import Database, fill_table
from .ingest import update_local_data, SFDATA
from .util import convert_legals, get_usd

ORACLE_TABLE_DEFNS = {
    "oracle_id": {"dtype": "CHAR(36)",
                  "primary": True},
    "name": {"dtype": "TEXT"},
    "mana_cost": {"dtype": "INT"},
    "cmc": {"dtype": "INT"},
    "type_line": {"dtype": "TEXT"},
    "oracle_text": {"dtype": "TEXT"},
    "rarity": {"dtype": "TEXT"},
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
        raise Exception(f"Table {table_name} doesn't exist")
    db.set_defns(PRICE_TABLE_DEFNS)
    # Get updated oracle cards data from scryfall
    fout = update_local_data()
    # Add time column
    today = datetime.datetime.now()
    date = f"{today.year}-{today.month}-{today.day}"
    fill_kw = {"date": date}
    # Fill table with data
    fill_table(db, f"bulk_data/{fout}.json", "prices", **fill_kw)
