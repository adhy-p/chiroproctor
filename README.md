# Chiroproctor

Chiroproctor - An IOT solution to help people maintain good postures. 

Using data collected from accelerometer and gyroscope, the ML model (LSTM + Dense network) predicts the user's current posture and send a notification to the user if he/she stay in a bad posture for some time.

BLE is used for SensorTag to Gateway communication and MQTT is used for Gateway to Cloud communication.

Click [here](https://docs.google.com/document/d/12bPJpNr083y5-u7QQZ1OwFSQWBxRyB-N2iGeVP8mYaQ/edit?usp=sharing) to see the architecture and design decision

Click [here](https://youtu.be/NwLOR8ldsmY) to see the demo

## Team members:
- Aldo Maximillian Sugito
- Ivander Jonathan Marella Waskito
- Justin Tzuriel Krisnahadi
- Hubertus Adhy Pratama Setiawan

## Running the project

### Setting up the environment
On the cloud and the gateway, run:
```
python3 install -r requirements.txt
```
### Running the code
On the cloud, run:
```
python3 model.py
```
On the gateway, run:
```
collect_data.sh
```
