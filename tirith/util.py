
# Standard Library
import os
import json
import struct
import datetime
from pathlib import Path


from functools import wraps


LEGALITIES = [
  'standard',
  'future',
  'historic',
  'timeless',
  'gladiator',
  'pioneer',
  'explorer',
  'modern',
  'legacy',
  'pauper',
  'vintage',
  'penny',
  'commander',
  'oathbreaker',
  'standardbrawl',
  'brawl',
  'alchemy',
  'paupercommander',
  'duel',
  'oldschool',
  'premodern',
  'predh',
]


PLAY_BOOSTER = {
    "cards" : {
        "num": 14,
        "common": 6,
        "uncommon": 3,
        "raythic": 1,
        "land": 1,
        "wildcard": 2,
        "misc": 1,
    },
    "odds" : {
        "rare": 87.5/100.0,
        "mythic": 12.5/100.0,
    }
}

# Standard legal setids
STANDARD_SETS = {
    # "mat": "6727e43d-31b6-45b0-ae05-7a811ba72f70",
    # "mom": "392f7315-dc53-40a3-a2cc-5482dbd498b3",
    # "one": "04bef644-343f-4230-95ee-255f29aa67a2",
    "bro": "4219a14e-6701-4ddd-a185-21dc054ab19b",
    "dmu": "4e47a6cd-cdeb-4b0f-8f24-cfe1a0127cb3",
    "woe": "79139661-13ee-43c4-8bad-a8c069f1a1df",
    "lci": "70169a6e-89d1-4a3a-aef7-3152958d55ac",
    "otj": "55a85ebe-644e-4bef-8be8-5290408be3d1",
    "mkm": "2b17794b-15c3-4796-ad6f-0887a0eceeca",
}

BLOOMBURROW_SETID = "a2f58272-bba6-439d-871e-7a46686ac018"

# Get absolute parent path to __file__
FDIR = Path(__file__).absolute().parent.parent


def ensure_cwd(*a, **kw):
    # Get desired dir
    ddir = kw.pop("ddir", None)
    def wrap(func):
        @wraps(func)
        def func_wrapper(*a, **kw):
            # Store cwd
            cwd = os.getcwd()
            # Go to ddir
            if ddir:
                os.chdir(ddir)
            # Run func
            fout = func(*a, **kw)
            # Return back to cwd
            if ddir:
                os.chdir(cwd)
            return fout
        return func_wrapper
    return wrap


def read_json(fjson):
    if os.path.isfile(fjson):
        with open(fjson) as fp:
            # Read what we just got and extract url
            jout = json.load(fp)
        return jout
    else:
        raise FileExistsError(f"{fjson} does not exist")
    

def get_usd(prices):
    v = prices.get("usd")
    if v:
        v = float(v)
    return v


def convert_legals(legalities):
    # Probably a better way to do this
    bitlist = []
    for leg in LEGALITIES:
        v = legalities[leg]
        if v == "not_legal":
            bitlist += [False]
        else:
            bitlist += [True]
    # Pack the booleans into a byte array
    packed_data = bytearray(struct.pack('?' * len(bitlist), *bitlist))
    return packed_data


def convert_blegals(blegalities):
    num_bits = len(blegalities)
    unpacked_data = []
    for i in range(num_bits):
        unpacked_data.append(struct.unpack('?', blegalities[i // 8])[0] & (1 << (i % 8)) != 0)
    return unpacked_data

def get_date():
    today = datetime.datetime.now()
    date_str = f"{today.year}-{today.month}-{today.day}"
    return date_str


