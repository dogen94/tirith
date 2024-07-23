import tirith.db
from tirith.tables import update_price_table
from tirith.ingest import update_local_data

import matplotlib.pyplot as plt
import numpy as np


# Murders at Karlov Manor set id
setid = "2b17794b-15c3-4796-ad6f-0887a0eceeca"

def estimate_set(setid, nsim=10000):
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
    # Read in prices db
    pdb = tirith.db.Database("db/oracle-prices.db")
    # Read in oracle-db
    db = tirith.db.Database("db/oracle-card.db")
    # Get oracle id for only those cards in the set of interest
    exec_str = "SELECT oracle_id FROM prices WHERE set_id=" + f"'{setid}'" 
    pdb.cursor.execute(exec_str)
    set_cids = pdb.cursor.fetchall()
    # Get rarity of these cards
    exec_str = "SELECT oracle_id,rarity,price_usd FROM cards WHERE set_id=" + f"'{setid}'" 
    db.cursor.execute(exec_str)
    # Build numpy array
    set_raritys = db.cursor.fetchall()
    # This is gross but jsut for now
    commons = []
    uncommons = []
    rares = [] 
    mythics = []
    for card in set_raritys:
        if card[1] == 'common':
            commons.append([*card])
        elif card[1] == 'uncommon':
            uncommons.append([*card])
        elif card[1] == 'rare':
            rares.append([*card])
        elif card[1] == 'mythic':
            mythics.append([*card])
    allcards = commons + uncommons + rares + mythics
    commons = np.array(commons)
    uncommons = np.array(uncommons)
    rares = np.array(rares)
    mythics = np.array(mythics)
    allcards = np.array(allcards)

    def sim_pack(commons, uncommons, rares, mythics, allcards):
        draws = []
        comdraws = np.random.choice(commons[:,2], 6)
        ucomdraws = np.random.choice(uncommons[:,2], 3)
        rare_or_mythic = np.random.uniform(0, 1)
        if rare_or_mythic > 87.5:
            rmdraws = np.random.choice(mythics[:,2], 1)
        else:
            rmdraws = np.random.choice(rares[:,2], 1)
        wildcards = np.random.choice(allcards[:,2], 2, replace=True)
        price = 0.0
        for card in [*comdraws, *ucomdraws, *rmdraws, *wildcards]:
            if card == None:
                card = 1.0;
            price+= float(card) 
        return price
    
    prices = np.empty(nsim)
    for i in range(nsim):
        prices[i] = sim_pack(commons, uncommons, rares, mythics, allcards)

    expectation_per_booster = np.mean(prices)
    expectation_per_bbox = expectation_per_booster * 36.0
    return expectation_per_bbox


if __name__ == "__main__":
    estimate_set(setid)
    # _ = update_local_data(force=True)
    # update_price_table()