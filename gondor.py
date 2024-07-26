import tensorflow as tf
import pandas as pd
import numpy as np
import keras
from sklearn.preprocessing import OneHotEncoder
import tirith.db


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


# Murders at Karlov Manor set id
setid = "2b17794b-15c3-4796-ad6f-0887a0eceeca"

# Try basic neural network
def example_dnn():
    # Read in oracle-db
    db = tirith.db.Database("db/oracle-card.db")
    
    # Get rarity of these cards
    exec_str = "SELECT oracle_id,rarity,price_usd FROM cards WHERE set_id=" + f"'{setid}'" 
    db.cursor.execute(exec_str)


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


example_dnn()
