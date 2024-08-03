import tensorflow as tf
import tensorflow_decision_forests as tfdf
import pandas as pd
import numpy as np
import keras
from sklearn.preprocessing import OneHotEncoder
import tirith.db
from tirith.util import STANDARD_SETS
from tirith.tables import ORACLE_TABLE_DEFNS


# Explore the data
def feature_selection():
    # Read in oracle-db
    db = tirith.db.Database("db/oracle-card.db")
    # Values of set ids
    setids = []
    setq = []
    for set,setid in STANDARD_SETS.items():
        setids.append(setid)
        setq.append("?")
    setids = tuple(setids)
    # Build tuple of question marks to match all setids
    setq = ",".join(setq)
    # Get rarity of these cards
    exec_str = "SELECT * FROM cards WHERE set_id IN (" + setq + ")" 
    db.cursor.execute(exec_str, setids)
    data = db.cursor.fetchall()
    db.set_defns(ORACLE_TABLE_DEFNS)
    data_arr = np.array(data)

    # Feature investigations
    # Is there a strongest color
    colors = data_arr[:,2]
    prices = data_arr[:,9]
    I = np.zeros((len(colors), 5), dtype=bool)
    for i,card in enumerate(colors):
        if card == '' or card is None:
            continue
        for j,color in enumerate(("R", "G", "B", "W", "U")):
            if color in card:
                I[i,j] = 1
        if prices[i] is None:
            prices[i] = np.NAN

    rmean = np.nanmean(prices[I[:,0]])
    gmean = np.nanmean(prices[I[:,1]])
    bmean = np.nanmean(prices[I[:,2]])
    wmean = np.nanmean(prices[I[:,3]])
    umean = np.nanmean(prices[I[:,4]])

    plt.scatter(np.arange(5), (rmean,gmean,bmean,wmean,umean))
    plt.show()

    plt.hist(prices[I[:,0]], bins=30, color='red', alpha=0.3)
    plt.hist(prices[I[:,1]], bins=30, color='green', alpha=0.3)
    plt.hist(prices[I[:,2]], bins=30, color='black', alpha=0.3)
    plt.hist(prices[I[:,3]], bins=30, color='yellow', alpha=0.3)
    plt.hist(prices[I[:,4]], bins=30, color='blue', alpha=0.3)
    plt.show()


GBTM_INPUTS = [
    "mana_cost",
    "cmc",
    "type_line",
    "colors",
    "power",
    "toughness",
    "rarity",
]

GBTM_OUT = "price_usd"

# Try basic neural network
def train_gbtm():
    # Read in oracle-db
    db = tirith.db.Database("db/oracle-card.db")
    db.set_defns(ORACLE_TABLE_DEFNS)
    # Values of set ids
    setids = []
    setq = []
    for set,setid in STANDARD_SETS.items():
        setids.append(setid)
        setq.append("?")
    setids = tuple(setids)
    # Build tuple of question marks to match all setids
    setq = ",".join(setq)
    # Build sql query string
    cols = ",".join(GBTM_INPUTS) + "," + GBTM_OUT
    # Get desired cols of these cards
    exec_str = "SELECT " + "%s "% cols + "FROM cards WHERE set_id IN (" + setq + ")" 
    db.cursor.execute(exec_str, setids)
    data = db.cursor.fetchall()
    # Build numpy array out of data
    data_arr = np.array(data)
    # Mask to remove any data w/o prices
    Iprice = np.ones(data_arr.shape[0], dtype=bool)
    # Preprocess the data
    for i in range(data_arr.shape[0]):
        # Sanitize prices
        if isinstance(data_arr[i, 7], type(None)):
            Iprice[i] = False
        # Sanitize mana_cost
        # Remove the brackets in the mana_cost and sort to remove uniqueness
        if isinstance(data_arr[i, 0], str):
            tmp = data_arr[i, 0].replace("{","").replace("}","")
            data_arr[i, 0] = "".join(sorted(tmp))
        # Set None mana_cost to 0
        elif isinstance(data_arr[i, 0], type(None)):
            data_arr[i, 0] = "0"

        # Sanitize colors
        # Remove the comma in the colors and sort to remove uniqueness
        if isinstance(data_arr[i, 3], str):
            tmp = data_arr[i, 3].replace(",","")
            data_arr[i, 3] = "".join(sorted(tmp))
        # Set None mana_cost to 0
        elif isinstance(data_arr[i, 3], type(None)):
            r""" Maybe create seperate land and artifacts here?"""
            data_arr[i, 3] = "0"

        # Sanitize power
        # Set * to -1 and None to -2
        if isinstance(data_arr[i, 4], type(None)):
            data_arr[i, 4] = -2
        elif isinstance(data_arr[i, 4], str):
            if "*" in data_arr[i, 4]:
                data_arr[i, 4] = -1

        # Sanitize toughness
        # Set * to -1 and None to -2
        if isinstance(data_arr[i, 5], type(None)):
            data_arr[i, 5] = -2
        elif isinstance(data_arr[i, 5], str):
            if "*" in data_arr[i, 5]:
                data_arr[i, 5] = -1
        
        # Sanitize prices
        if isinstance(data_arr[i, 7], type(None)):
            data_arr[i, 5] = np.NaN

    # Number of trials in autotuner
    TUNER_TRIALS=100
    all_cols = [*GBTM_INPUTS, GBTM_OUT]
    train_data_df0 = pd.DataFrame(data_arr[Iprice,:], columns=all_cols)
    # Fix dtypes
    train_data_df = train_data_df0.convert_dtypes()

    # # Save the preprocessed data for import next time
    # train_data_df.to_csv()

    # Split data set to test and training data
    def split_dataset(dataset, test_ratio=0.35):
        test_indices = np.random.rand(len(dataset)) < test_ratio
        return dataset[~test_indices], dataset[test_indices]

    # Get split dataframe (65/35)
    train_ds_pd, valid_test_ds_pd = split_dataset(train_data_df)
    # Split valid and test (25/10)
    valid_ds_pd, test_ds_pd = split_dataset(train_data_df, test_ratio=0.285)

    # Loading pd dataframe into tf dataset
    label = 'price_usd'
    train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label=label, task = tfdf.keras.Task.REGRESSION)
    valid_ds = tfdf.keras.pd_dataframe_to_tf_dataset(valid_ds_pd, label=label, task = tfdf.keras.Task.REGRESSION)

    # Set hyperparameter tuner
    tuner = tfdf.tuner.RandomSearch(num_trials=TUNER_TRIALS,
                                    use_predefined_hps=True,
                                    trial_num_threads=20)

    # Start with random forest model, Use top ranking hyper parameters
    rf = tfdf.keras.GradientBoostedTreesModel(
        num_threads=16,
        tuner=tuner,
        task = tfdf.keras.Task.REGRESSION)
    rf.compile(metrics=["mae"]) # Optional, you can use this to include a lis of eval metrics
    with tf.device('/device:GPU: 0'): 
        # Train model on gpu
        rf.fit(x=train_ds, verbose=1)

    # # Evaluate model on validation
    evaluation = rf.evaluate(x=valid_ds,return_dict=True)

    # Just save model for now
    tuning_logs = rf.make_inspector().tuning_logs()
    T = tuning_logs[tuning_logs.best].iloc[0]
    T.to_csv("models/autotune_gbmt-%itrials.csv" % TUNER_TRIALS)

    # Load test dataframe into keras ds
    test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(
        test_ds_pd,
        task = tfdf.keras.Task.REGRESSION)
    # Predict the prices
    preds = rf.predict(test_ds)
    # Get actual prices of prediction
    test_prices = test_ds_pd["price_usd"].to_numpy()
    # Mean square error
    mse = np.mean((preds - test_prices)**2) / len(test_prices)
    print(mse)

# feature_selection()
train_gbtm()
