# import tensorflow as tf
import pandas as pd
import numpy as np
# import keras
# from sklearn.preprocessing import OneHotEncoder
import tirith.db
from tirith.util import STANDARD_SETS
from tirith.tables import ORACLE_TABLE_DEFNS


def build_and_compile_model():
    model = keras.Sequential([
        tf.keras.layers.Dense(10000, activation='relu',
                        kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dense(10000, activation='relu',
                        kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dense(10000, activation='relu',
                        kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dense(1)
    ])

    model.compile(loss='mean_absolute_error',
                    optimizer=tf.keras.optimizers.Adam(0.001))
    return model



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

# Try basic neural network
def example_dnn():
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

    # Random feature investigations

    # Read in training data
    train_data_dir = "data/train.csv"
    train_data_df = pd.read_csv(train_data_dir)

    # Drop id column
    train_data_df = train_data_df.drop("Id", axis=1)


    # Split data set to test and training data
    def split_dataset(dataset, test_ratio=0.30):
        test_indices = np.random.rand(len(dataset)) < test_ratio
        return dataset[~test_indices], dataset[test_indices]

    # Get split dataframe
    train_ds_pd = train_data_df


    enc = OneHotEncoder(handle_unknown='ignore')

    # Post process data, only keep numbers
    num_cols = []
    obj_cols = []
    for k in train_ds_pd.dtypes.keys():
        dt = train_ds_pd.dtypes[k]
        if (dt == 'int64' or dt == 'float64'):
            num_cols.append(k)
        else:
            obj_cols.append(k)

    # Preprocess numerical data 
    # Set NANs to mean of data
    train_np = train_ds_pd[num_cols].to_numpy()
    for i in range(train_np.shape[-1]):
        mean = np.nanmean(train_np[:,i])
        train_np[:,i] = np.nan_to_num(train_np[:,i], mean)

    # Preprocess non-numerical data
    train_obj = train_ds_pd[obj_cols].to_numpy()
    enc.fit(train_obj)
    train_obj_np = enc.transform(train_obj).toarray()
    n_obj = train_obj_np.shape[-1]
    # Combine preprocessed data of all types
    train_all = np.hstack([train_obj_np, train_np])

    callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='auto',
        restore_best_weights=True,
        start_from_epoch=0
    )

    dnn = build_and_compile_model()
    with tf.device('/device:GPU: 0'): 
        # Train model on gpu
        dnn.fit(train_all[:,:-1], train_all[:,-1], epochs=200, shuffle=True,
                validation_split=0.25, callbacks=[callback])


    # Save model
    dnn.save('models/dnn_set3_mae.keras')
    # Submission prediction
    test_file_path = "data/test.csv"
    test_data = pd.read_csv(test_file_path)
    ids = test_data.pop('Id')

    # Set NANs to mean of data
    test_np = test_data[num_cols[:-1]].to_numpy()
    for i in range(test_np.shape[-1]-1):
        mean = np.nanmean(test_np[:,i])
        test_np[:,i] = np.nan_to_num(test_np[:,i], mean)

    test_obj = test_data[obj_cols].to_numpy()
    test_obj_np = enc.transform(test_obj).toarray()
    test_all = np.hstack([test_obj_np, test_np])


    def loss(target_y, predicted_y):
        return tf.reduce_mean(tf.square(target_y - predicted_y))

    def maeloss(target_y, predicted_y):
        return tf.reduce_mean(tf.abs(target_y - predicted_y))

    print("Current loss: %1.6f" % loss(dnn(train_all[:,:-1]), train_all[:,-1]).numpy())
    
    preds = dnn.predict(test_all)


feature_selection()
# example_dnn()
