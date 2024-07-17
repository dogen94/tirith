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
    "legalities": {"dtype": "CHAR(22)",
                   "func": convert_legals},
    "set_id": {"dtype": "CHAR(36)"},
    "prices": {"dtype": "FLOAT",
               "primary": True,
               "func": get_usd,
               "trans": "price_usd"},
}

