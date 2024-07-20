import tirith.db
from tirith.tables import update_price_table
from tirith.ingest import update_local_data

import matplotlib.pyplot as plt

def estimate_set():
    r"""wait this is gross, they keep changing all this stuff
    So a "play booster box" is 36 packs that's standard
    Then you need to estimate the value of each pack;
    The number of cards in each pack changes. For newest stuff it's 14 cards;
    Those 14 cards have {
        6 commons,
        3 uncommon,
        1 rare/mythic,
        1 land,
        1 non-foil wildcard [any rarity],
        1 foil wildcard [any rarity],
        1 token/art/ad card
    }
    The wildcards will just be the expected value of the entire set. Will need
    to actually track the foil prices too if I want to do this fully. Could proxy
    it with just a price multiplier on the non-foil like 1.1x (its smaller than 
    I thought). Probably just ignore the land value or just put it static 0.25$.
    Then I need the odds for the commons,uncommon,rare,mythics.
    """



if __name__ == "__main__":
    _ = update_local_data(force=True)
    update_price_table()