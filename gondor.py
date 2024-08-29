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
    "oracle_id",
]

GBTM_OUT = "price_usd"

# Try basic decision forest
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
        if isinstance(data_arr[i, -1], type(None)):
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
        
    # Number of trials in autotuner
    TUNER_TRIALS=100
    all_cols = [*GBTM_INPUTS, GBTM_OUT]
    train_data_df0 = pd.DataFrame(data_arr[Iprice,:], columns=all_cols)
    # Remove oracle_id
    train_data_df0.pop("oracle_id")
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

    return rf
    """
    # # Evaluate model on validation÷turn_dict=True)
    setq = ["?"]
    setids = tuple( [tar_setid] )
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
    Iprice1 = np.ones(data_arr.shape[0], dtype=bool)
    # Preprocess the data
    for i in range(data_arr.shape[0]):
        # Sanitize prices
        if isinstance(data_arr[i, 7], type(None)):
            Iprice1[i] = False

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
            # Maybe create seperate land and artifacts here?
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

    test_prices = data_arr[:,-1]

    all_cols = [*GBTM_INPUTS, GBTM_OUT]
    test_data_df0 = pd.DataFrame(data_arr, columns=all_cols)
    # Fix dtypes
    test_data_df = test_data_df0.convert_dtypes()
    test_data_df.pop("price_usd")

    # Load test dataframe into keras ds
    test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(
        test_data_df,
        task = tfdf.keras.Task.REGRESSION)
    # Predict the prices
    preds = rf.predict(test_ds)
    # Get actual prices of prediction
    # test_prices = test_data_df["price_usd"].to_numpy()
    # Mean square error
    mse = np.mean((preds[Iprice1,0] - test_prices[Iprice1])**2) / len(test_prices[Iprice1])
    print(mse)
    """

def predict_gbtm(set_tars, gbtm):
    # Read in oracle-db
    db = tirith.db.Database("db/oracle-card.db")
    db.set_defns(ORACLE_TABLE_DEFNS)
    # Values of set ids
    setids = []
    setq = []
    # Get bloomburrow data
    for set,setid in set_tars.items():
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
    
    # Preprocess the data
    for i in range(data_arr.shape[0]):
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
    train_data_df0 = pd.DataFrame(data_arr, columns=all_cols)
    # Fix dtypes
    train_data_df = train_data_df0.convert_dtypes()

    # # Save the preprocessed data for import next time
    # train_data_df.to_csv()


    # Loading pd dataframe into tf dataset
    label = 'price_usd'
    train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds_pd, label=label, task = tfdf.keras.Task.REGRESSION)
    valid_ds = tfdf.keras.pd_dataframe_to_tf_dataset(valid_ds_pd, label=label, task = tfdf.keras.Task.REGRESSION)


    "models/autotune_gbmt-%itrials.csv" % TUNER_TRIALS


    # Start with random forest model, Use top ranking hyper parameters
    rf = tfdf.keras.GradientBoostedTreesModel(
        num_threads=16,
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
    

TP_INPUTS = [
    "type_line",
    "oracle_text",
    "name",
    "oracle_id",
]

TP_OUT = "price_usd"

def text_regression(model, tokenizer):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error

    # Read in db
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
    cols = ",".join(TP_INPUTS) + "," + TP_OUT
    # Get desired cols of these cards
    exec_str = "SELECT " + "%s "% cols + "FROM cards WHERE set_id IN (" + setq + ")" 
    db.cursor.execute(exec_str, setids)
    data = db.cursor.fetchall()
    # Build numpy array out of data
    data_arr = np.array(data)
    
    I = np.ones(data_arr.shape[0], dtype=bool)
    for icase in range(data_arr.shape[0]):
        if data_arr[icase,0] == None:
            data_arr[icase,0] = ""
        if data_arr[icase,1] == None:
            data_arr[icase,1] = ""
        if data_arr[icase,-1] == None or data_arr[icase,-1] == "":
            I[icase] = False
    missing_inds = np.where(I == False)[0]
    # Combine two texts
    data_arr[:, 0] = data_arr[:, 0] + ":: " + data_arr[:, 1]
    # Sample data
    texts = data_arr[I, 0] + ":: " + data_arr[I, 1]
    values = data_arr[I, -1]

    # Preprocessing and feature extraction
    X = []
    for tex in texts:
        inputs = tokenizer(tex, return_tensors="pt")
        out = model(**inputs)
        X.append(out.pooler_output[0].detach().numpy())
    # X = vectorizer.fit_transform(texts)
    X_arr = np.array(X)
    y = values

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_arr, y, test_size=0.2, random_state=42)

    # Train a model
    token_model = LinearRegression()
    token_model.fit(X_train, y_train)
    

    return token_model
    """
    # Predict and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    # Predict values of missing text
    new_texts = data_arr[missing_inds,0] + ":: " + data_arr[missing_inds,1]
    new_X = vectorizer.transform(new_texts)
    predicted_values = model.predict(new_X)
    print(f'Predicted Value: {predicted_values}')
    """

# both of these need to take in the same card and mix predictions


def predict_model(model, tokenizer, tar_setid=tirith.util.BLOOMBURROW_SETID):
    gbtm_model = train_gbtm()
    token_model = text_regression(model, tokenizer)

    # Read in db for token model, text not preprocessed yet :(
    db = tirith.db.Database("db/oracle-card.db")
    db.set_defns(ORACLE_TABLE_DEFNS)

    # # Evaluate model on validation÷turn_dict=True)
    setq = ["?"]
    setids = tuple( [tar_setid] )
    # Build tuple of question marks to match all setids
    setq = ",".join(setq)
    # Build sql query string
    cols = ",".join(TP_INPUTS) + "," + TP_OUT
    # Get desired cols of these cards
    exec_str = "SELECT " + "%s "% cols + "FROM cards WHERE set_id IN (" + setq + ")" 
    db.cursor.execute(exec_str, setids)
    data = db.cursor.fetchall()
    # Build numpy array out of data
    text_data_arr = np.array(data)
    # Process data
    I = np.ones(text_data_arr.shape[0], dtype=bool)
    for icase in range(text_data_arr.shape[0]):
        if text_data_arr[icase,0] == None:
            text_data_arr[icase,0] = ""
        if text_data_arr[icase,1] == None:
            text_data_arr[icase,1] = ""
        if text_data_arr[icase,-1] == None or text_data_arr[icase,-1] == "":
            I[icase] = False
    missing_inds = np.where(I == False)[0]
    # Combine two texts
    text_data_arr[:, 0] = text_data_arr[:, 0] + ":: " + text_data_arr[:, 1]
    # Sample data
    texts = text_data_arr[I, 0] + ":: " + text_data_arr[I, 1]
    values = text_data_arr[I, -1]
    # Preprocessing and feature extraction
    X = []
    for tex in texts:
        inputs = tokenizer(tex, return_tensors="pt")
        out = model(**inputs)
        X.append(out.pooler_output[0].detach().numpy())
    transformed_texts = np.array(X)

    # Read in db
    db = tirith.db.Database("db/oracle-card-pp3.db")
    db.set_defns(ORACLE_TABLE_DEFNS)

    setq = ["?"]
    setids = tuple([tar_setid])
    # Build tuple of question marks to match all setids
    setq = ",".join(setq)
    # Build sql query string
    cols = ",".join(GBTM_INPUTS) + "," + GBTM_OUT
    # Get desired cols of these cards
    exec_str = "SELECT " + "%s "% cols + "FROM cards WHERE set_id IN (" + setq + ")" 
    db.cursor.execute(exec_str, setids)
    data = db.cursor.fetchall()
    # Build numpy array out of data
    gbtm_data_arr = np.array(data)

    all_cols = [*GBTM_INPUTS, GBTM_OUT]
    pred_data_df0 = pd.DataFrame(gbtm_data_arr, columns=all_cols)
    # Fix dtypes
    pred_data_df = pred_data_df0.convert_dtypes()
    # Manual fix mana_cost col
    pred_data_df["mana_cost"] = pred_data_df["mana_cost"].astype(str)
    pred_data_df.pop("price_usd")
    pred_data_df.pop("oracle_id")    
    # Load prediction dataframe into keras ds
    pred_ds = tfdf.keras.pd_dataframe_to_tf_dataset(
        pred_data_df,
        task = tfdf.keras.Task.REGRESSION)
    
    # Predict the prices
    gbtm_prediction = gbtm_model.predict(pred_ds)
    token_prediction = token_model.predict(transformed_texts)

    plt.plot(gbtm_prediction, c="r")
    plt.plot(token_prediction, c="b")
    # plt.plot((1-0.1)*gbtm_prediction[I,0] + 0.1*token_prediction, c="r")
    plt.plot(values, c="k")
    plt.show()
    
    print("wait")


from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")
predict_model(model, tokenizer)