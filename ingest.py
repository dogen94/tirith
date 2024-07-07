
# Standard Library
import os
import datetime
from pathlib import Path
from subprocess import Popen,PIPE

# Local imports
from util import ensure_cwd


# Scryfall data list
SFDATA = [
    "oracle-cards"
]
# Scryfall address
SFADDR = 'https://api.scryfall.com/bulk-data/'
# Get absolute parent path to __file__
FDIR = Path(__file__).absolute().parent

@ensure_cwd(ddir=FDIR)
# Get scryfall data
def ingest_url(url, **kw):
    # Get kws
    fout = kw.pop("fout", None)
    # Build command line
    cmdi = ["curl", f"{url}"]
    if fout:
        cmdi += ["-o", f"{fout}"]
    # Enable stderr
    cmdi += ["--fail", "--silent", "--show-error"]
    # Run process
    process = Popen(cmdi, stdout=PIPE, stderr=PIPE, encoding="UTF-8")
    _, stderr = process.communicate()
    # Raise stderr exception if exists
    if stderr:
        raise Exception(stderr)


def update_data(data_list=SFDATA, force=False):
    time = datetime.datetime.now()
    for data in data_list:
        fout = f"{data}-{time.year}-{time.month}-{time.day}"
        # Dont ingest if exists and no force, else ingest
        if not(os.path.exists(fout)) and force:
            ingest_url(SFADDR + f"{data}", fout=f"bulk_data/{fout}")
