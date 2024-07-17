
# Standard Library
import os
import json
import struct
import datetime

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


def ensure_cwd(*a, **kw):
    def wrap(func):
        @wraps(func)
        def func_wrapper(*a, **kw):
            # Store cwd
            cwd = os.getcwd()
            # Get desired dir
            ddir = kw.pop("ddir", None)
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


