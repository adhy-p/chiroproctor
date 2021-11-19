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
currStatus = ""
badCounter = 0
consecBad = 0
consecGood = 0
battery = None

def create_and_train_model():
    df1 = pd.read_csv("training_data/multitrain1.csv")
    df2 = pd.read_csv("training_data/multitrain2.csv")

    df1 = df1.drop(columns=['time', 'posture'])
    df2 = df2.drop(columns=['time'])

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
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs = 150, validation_split = 0.20, batch_size = 200, verbose = 1)
    loss, accuracy = model.evaluate(X_test, y_test, batch_size = 200, verbose = 1)
    print("Test Accuracy :", accuracy)
    print("Test Loss :", loss)
    return model

# def test_model_with_csv():
#     time_steps = 10
#     df_test = pd.read_csv("adhy-sitting-test.csv")
#     print(df_test.head())
#     segments_test = []
#     for i in range(0,  df_test.shape[0] - time_steps, 2):  
#         xs = df_test['acc_x'].values[i: i + 10]
#         ys = df_test['acc_y'].values[i: i + 10]
#         zs = df_test['acc_z'].values[i: i + 10]
#         segments_test.append([xs, ys, zs])

#     reshaped_segments_test = np.asarray(segments_test, dtype = np.float32).reshape(-1, time_steps, 3)
#     predictions = model.predict(reshaped_segments_test)

#     for p in predictions:
#         arr  = ["KYPHOSIS", "LORDOSIS", "NORMAL", "SCOLIOSIS"]
#         print(arr[p.index(max(p))])

def predict_posture():
    global sensor_1_data
    global sensor_2_data
    num_data_points = min(len(sensor_1_data), len(sensor_2_data))
    if num_data_points == 0:
        return None

    df1 = pd.DataFrame(np.array(sensor_1_data[-1]["data"]))
    df2 = pd.DataFrame(np.array(sensor_2_data[-1]["data"]))
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

    sensor_1_data = []
    sensor_2_data = []

    global consecGood
    global consecBad
    global badCounter

    for p in predictions:
        arr  = ["KYPHOSIS", "LORDOSIS", "NORMAL", "SCOLIOSIS"]
        result = arr[np.argmax(p)] 
        print(p, result)
        if result != "NORMAL":
            consecGood = 0
            consecBad += 1
            if consecBad > 10:
                badCounter += 1
        else:
            consecBad = 0
            consecGood += 1
            if consecGood > 10:
                badCounter = 0
        return {"prediction":result}


def get_video_recommendations():
    global badCounter
    global currStatus
    if badCounter < 10 and currStatus != "good":
        currStatus = "good"
        return {"videos": goodVideos}
    elif badCounter >= 10 and badCounter < 20 and currStatus != "medium":
        currStatus = "medium"
        return {"videos": mediumVideos}
    elif badCounter >= 20 and currStatus != "bad":
        currStatus = "bad"
        return {"videos": badVideos}
    return None

def send_data(client, data, topic, retain):
    print("Sending data: ", data)
    client.publish(topic, json.dumps(data), retain=retain)

def on_prediction_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Successfully connected to broker.")
        client.subscribe("Group_2/classify")
    else:
        print("Connection failed with code: %d." % rc)

def on_battery_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Successfully connected to broker.")
        client.subscribe("Group_2/battery_info") # from Gateway
    else:
        print("Connection failed with code: %d." % rc)

def on_prediction_message(client, userdata, msg):
    recv_dict = json.loads(msg.payload)
    if recv_dict["sensortag_id"] == 1:
        sensor_1_data.append(recv_dict)
    else:
        sensor_2_data.append(recv_dict)
    posture_data = np.array(recv_dict["data"])

# acts like a proxy and simply forwards the data
def on_battery_message(client, userdata, msg):
    recv_dict = json.loads(msg.payload)
    send_data(client, recv_dict, "Group_2/battery", True)

def setup(hostname, connect_callback, message_callback):
    client = mqtt.Client()
    client.on_connect = connect_callback
    client.on_message = message_callback
    client.connect(hostname)
    client.loop_start()
    return client

def main():
    global model
    model = create_and_train_model()
    # test_model_with_csv()
    gateway = setup("127.0.0.1", on_prediction_connect, on_prediction_message)
    gateway_metadata = setup("127.0.0.1", on_battery_connect, on_battery_message)
    while True:
        time.sleep(0.5)
        result = predict_posture()
        if result is not None:
            send_data(gateway, result, "Group_2/predict", False)
        videos = get_video_recommendations()
        if videos is not None:
            send_data(gateway, videos, "Group_2/video", True)
        pass

if __name__ == '__main__':
    main()