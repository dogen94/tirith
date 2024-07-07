
# Standard Library
import os
import datetime
from pathlib import Path
from subprocess import Popen,PIPE

# Local imports
from util import ensure_cwd


SF_ADR = "https://api.scryfall.com/bulk-data/"

FDIR = Path(__file__).absolute().parent

@ensure_cwd(ddir=FDIR)
# Get scryfall data
def ingest_url(url, **kw):
    # Get kws
    odir = kw.pop("odir", None)
    # Build command line
    cmdi = ["curl", f"{url}"]
    if odir:
        cmdi += ["-o", f"{odir}"]
    # Run process
    process = Popen(cmdi, stdout=PIPE, stderr=PIPE)
    stdout, stderr = process.communicate()


time = datetime.datetime.now()
fout = f"oracle-cards-{time.year}-{time.month}-{time.day}"
ingest_url(SF_ADR + "oracle-cards", odir=f"bulk_data/{fout}")