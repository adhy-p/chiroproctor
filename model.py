import json
import paho.mqtt.client as mqtt
import pandas as pd 
from scipy import stats
import numpy as np
from sklearn.model_selection import train_test_split
import time
from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten, Dropout


model = None
sensor_1_data = []
sensor_2_data = []
goodVideos = ["OyK0oE5rwFY", "FkdceBcRa5w", "nr-pHthhMBE"]
mediumVideos = ["dCsgXitfdls", "3aRpAO6bfvA"]
badVideos = ["RqcOCBb4arc", "5R54QoUbbow"]
badCounter = 0
consecBad = 0
consecGood = 0

def create_and_train_model():
    df1 = pd.read_csv("test1.csv")
    df2 = pd.read_csv("test2.csv")

    df1 = df1.iloc[6:].reset_index()
    df2 = df2.iloc[2:].reset_index()

    df1 = df1.drop(columns=['index', 'time', 'posture'])
    df2 = df2.drop(columns=['index', 'time'])

    df1.columns = [str(col) + '_1' for col in df1.columns]
    df2.columns = [str(col) + '_2' if not col == 'posture' else 'posture' for col in df2.columns]

    df = pd.concat([df1, df2], axis=1).dropna()

    segments = []
    labels = []
    time_steps = 10

    for i in range(0,  df.shape[0] - time_steps, 2):  
        ax1 = df['acc_x_1'].values[i: i + 10]
        ay1 = df['acc_y_1'].values[i: i + 10]
        az1 = df['acc_z_1'].values[i: i + 10]
        ax2 = df['acc_x_2'].values[i: i + 10]
        ay2 = df['acc_y_2'].values[i: i + 10]
        az2 = df['acc_z_2'].values[i: i + 10]
        gx1 = df['gyr_x_1'].values[i: i + 10]
        gy1 = df['gyr_y_1'].values[i: i + 10]
        gz1 = df['gyr_z_1'].values[i: i + 10]
        gx2 = df['gyr_x_2'].values[i: i + 10]
        gy2 = df['gyr_y_2'].values[i: i + 10]
        gz2 = df['gyr_z_2'].values[i: i + 10]
        label = stats.mode(df['posture'][i: i + 10])[0][0]
        segments.append([ax1, ay1, az1, ax2, ay2, az2])
        labels.append(label)

    reshaped_segments = np.asarray(segments, dtype = np.float32).reshape(-1, time_steps, 6)
    labels = np.asarray(pd.get_dummies(labels), dtype = np.float32)
    X_train, X_test, y_train, y_test = train_test_split(reshaped_segments, labels, test_size = 0.2, random_state = 0)

    model = Sequential()
    # RNN layer
    model.add(LSTM(units = 128, input_shape = (X_train.shape[1], X_train.shape[2])))
    # Dropout layer
    model.add(Dropout(0.5)) 
    # Dense layer with ReLu
    model.add(Dense(units = 100, activation='relu'))
    # Softmax layer
    model.add(Dense(y_train.shape[1], activation = 'softmax'))
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs = 20, validation_split = 0.20, batch_size = 200, verbose = 1)
    loss, accuracy = model.evaluate(X_test, y_test, batch_size = 200, verbose = 1)
    print("Test Accuracy :", accuracy)
    print("Test Loss :", loss)
    return model

def test_model_with_csv():
    time_steps = 10
    df_test = pd.read_csv("adhy-sitting-test.csv")
    print(df_test.head())
    segments_test = []
    for i in range(0,  df_test.shape[0] - time_steps, 2):  
        xs = df_test['acc_x'].values[i: i + 10]
        ys = df_test['acc_y'].values[i: i + 10]
        zs = df_test['acc_z'].values[i: i + 10]
        segments_test.append([xs, ys, zs])

    reshaped_segments_test = np.asarray(segments_test, dtype = np.float32).reshape(-1, time_steps, 3)
    predictions = model.predict(reshaped_segments_test)

    for p in predictions:
        if p[0] > p[1]:
            print("bad", p)
        else:
            print("good", p)

def predict_posture():
    num_data_points = min(len(sensor_1_data), len(sensor_2_data))
    if num_data_points == 0:
        return None

    df1 = pd.DataFrame(np.array(sensor_1_data[0]["data"]))
    df2 = pd.DataFrame(np.array(sensor_2_data[0]["data"]))
    df1 = df1.drop(columns=[6])
    df2 = df2.drop(columns=[6])

    df1.columns = [str(col) + '_1' for col in df1.columns]
    df2.columns = [str(col) + '_2' for col in df2.columns]

    df = pd.concat([df1, df2], axis=1).dropna()
    # print(df.columns)
    segments_test = []
    ax1 = df['0_1'].values
    ay1 = df["1_1"].values
    az1 = df["2_1"].values
    gx1 = df["3_1"].values
    gy1 = df["4_1"].values
    gz1 = df["5_1"].values
    ax2 = df["0_2"].values
    ay2 = df["1_2"].values
    az2 = df["2_2"].values
    gx2 = df["3_2"].values
    gy2 = df["4_2"].values
    gz2 = df["5_2"].values
    segments_test.append([ax1, ay1, az1, ax2, ay2, az2])

    reshaped_segments_test = np.asarray(segments_test, dtype = np.float32).reshape(-1, 10, 6)
    predictions = model.predict(reshaped_segments_test)

    sensor_1_data.pop(0)
    sensor_2_data.pop(0)

    for p in predictions:
        if p[0] > p[1]:
            print("bad", p)
            consecGood = 0
            consecBad += 1
            if consecBad > 10:
                badCounter += 1
            return {"prediction": "bad"}
        else:
            print("good", p)
            consecBad = 0
            consecGood += 1
            if consecGood > 10:
                badCounter = 0
            return {"prediction": "good"}

def get_video_recommendations():
    if badCounter < 30:
        return {"message": goodVideos}
    elif badCounter < 100:
        return {"message": mediumVideos}
    else:
        return {"message": badVideos}

def send_data(client, data, topic):
    print("Sending data: ", data)
    client.publish(topic, json.dumps(data))

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Successfully connected to broker.")
        client.subscribe("Group_2/classify")
    else:
        print("Connection failed with code: %d." % rc)

def on_message(client, userdata, msg):
    recv_dict = json.loads(msg.payload)
    if recv_dict["sensortag_id"] == 1:
        sensor_1_data.append(recv_dict)
    else:
        sensor_2_data.append(recv_dict)
    posture_data = np.array(recv_dict["data"])

def setup(hostname):
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(hostname)
    client.loop_start()
    return client

def main():
    global model
    model = create_and_train_model()
    # test_model_with_csv()
    client = setup("127.0.0.1")
    while True:
        time.sleep(0.5)
        result = predict_posture()
        send_data(client, result, "Group_2/predict")
        videos = get_video_recommendations()
        send_data(client, videos, "Group_2/video")
        pass

if __name__ == '__main__':
    main()