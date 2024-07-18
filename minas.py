import tirith.db
from tirith.tables import update_price_table
from tirith.ingest import update_local_data

import matplotlib.pyplot as plt

if __name__ == "__main__":
    _ = update_local_data(force=True)
    update_price_table()