from bluepy.btle import UUID, Peripheral, DefaultDelegate, AssignedNumbers
import csv
import datetime as dt
import json
import math
import os
import paho.mqtt.client as mqtt
import pandas as pd
import signal
import struct
import sys
from time import time

# For buzzer
consecutive_bad_postures = 0
current_posture = None
buzzer_is_active = False

# For data collection
i = 0
posture_label = "NORMAL"

def sigint_handler(sig, frame):
    global posture_label
    global i
    posture_states = ["NORMAL", "KYPHOSIS", "SCOLIOSIS", "LORDOSIS"]
    posture_label = posture_states[i]
    i += 1
    i %= 4
    # print('Posture state changed')
    # print(posture_states[i])
    
def sigterm_handler(sig, frame):
    sys.exit(0)

class CsvReader:
    headers = ['time', 'acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'posture']

    def __init__(self, sensortag_mac):
        sensortag_id = (1 if sensortag_mac == "54:6C:0E:B6:D3:85" else 2)
        self.filename = "sensortag-" + str(sensortag_id) + '.csv'
        self.file = open(self.filename, 'a')
        self.writer = csv.writer(self.file, lineterminator='\n')
        self.write_header()

    def write_header(self):
        self.writer.writerow(self.headers)

    def write_row(self, rowdata):
        timestamp = dt.datetime.now()
        print(timestamp)
        self.writer.writerow([timestamp] + list(rowdata))


class MQTT:
    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            print("Connected.")
            client.subscribe("Group_2/predict")
        else:
            print("Failed to connect. Error code: %d." % rc)

    def on_message(self, client, userdata, msg):
        result = json.loads(msg.payload)

        if result is None:
            return

        global current_posture
        if result["prediction"] == "NORMAL":
            current_posture = "good"
        else:
            current_posture = "bad"
        
    def setup(self, hostname):
        client = mqtt.Client()
        client.on_connect = self.on_connect
        client.on_message = self.on_message
        client.connect(hostname)
        client.loop_start()
        return client

    def send_data(self, client, sensortag_mac, number, data):
        sensortag_id = (1 if sensortag_mac == "54:6C:0E:B6:D3:85" else 2)
        send_dict = {"data_id":number, "sensortag_id": sensortag_id, "data":data, "posture": posture_label}
        client.publish("Group_2/classify", json.dumps(send_dict))

    def send_battery_data(self, client, sensortag_mac, data):
        sensortag_id = (1 if sensortag_mac == "54:6C:0E:B6:D3:85" else 2)
        send_dict = {"sensortag_id": sensortag_id, "battery": data}
        client.publish("Group_2/battery_info", json.dumps(send_dict))


def _TI_UUID(val):
    return UUID("%08X-0451-4000-b000-000000000000" % (0xF0000000+val))


# Sensortag versions
AUTODETECT = "-"
SENSORTAG_V1 = "v1"
SENSORTAG_2650 = "CC2650"


class SensorBase:
    # Derived classes should set: svcUUID, ctrlUUID, dataUUID
    sensorOn = struct.pack("B", 0x01)
    sensorOff = struct.pack("B", 0x00)

    def __init__(self, periph):
        self.periph = periph
        self.service = None
        self.ctrl = None
        self.data = None

    def enable(self):
        if self.service is None:
            self.service = self.periph.getServiceByUUID(self.svcUUID)
        if self.ctrl is None:
            self.ctrl = self.service.getCharacteristics(self.ctrlUUID)[0]
        if self.data is None:
            self.data = self.service.getCharacteristics(self.dataUUID)[0]
        if self.sensorOn is not None:
            self.ctrl.write(self.sensorOn, withResponse=True)

    def read(self):
        return self.data.read()

    def disable(self):
        if self.ctrl is not None:
            self.ctrl.write(self.sensorOff)

    # Derived class should implement _formatData()


def calcPoly(coeffs, x):
    return coeffs[0] + (coeffs[1]*x) + (coeffs[2]*x*x)


class AccelerometerSensor(SensorBase):
    svcUUID = _TI_UUID(0xAA10)
    dataUUID = _TI_UUID(0xAA11)
    ctrlUUID = _TI_UUID(0xAA12)

    def __init__(self, periph):
        SensorBase.__init__(self, periph)
        if periph.firmwareVersion.startswith("1.4 "):
            self.scale = 64.0
        else:
            self.scale = 16.0

    def read(self):
        '''Returns (x_accel, y_accel, z_accel) in units of g'''
        x_y_z = struct.unpack('bbb', self.data.read())
        return tuple([(val/self.scale) for val in x_y_z])


class MovementSensorMPU9250(SensorBase):
    svcUUID = _TI_UUID(0xAA80)
    dataUUID = _TI_UUID(0xAA81)
    ctrlUUID = _TI_UUID(0xAA82)
    sensorOn = None
    GYRO_XYZ = 7
    ACCEL_XYZ = 7 << 3
    MAG_XYZ = 1 << 6
    ACCEL_RANGE_2G = 0 << 8
    ACCEL_RANGE_4G = 1 << 8
    ACCEL_RANGE_8G = 2 << 8
    ACCEL_RANGE_16G = 3 << 8

    def __init__(self, periph):
        SensorBase.__init__(self, periph)
        self.ctrlBits = 0

    def enable(self, bits):
        SensorBase.enable(self)
        self.ctrlBits |= bits
        self.ctrl.write(struct.pack("<H", self.ctrlBits))

    def disable(self, bits):
        self.ctrlBits &= ~bits
        self.ctrl.write(struct.pack("<H", self.ctrlBits))

    def rawRead(self):
        dval = self.data.read()
        return struct.unpack("<hhhhhhhhh", dval)


class AccelerometerSensorMPU9250:
    def __init__(self, sensor_):
        self.sensor = sensor_
        self.bits = self.sensor.ACCEL_XYZ | self.sensor.ACCEL_RANGE_4G
        self.scale = 8.0/32768.0  # TODO: why not 4.0, as documented?

    def enable(self):
        self.sensor.enable(self.bits)

    def disable(self):
        self.sensor.disable(self.bits)

    def read(self):
        '''Returns (x_accel, y_accel, z_accel) in units of g'''
        rawVals = self.sensor.rawRead()[3:6]
        return tuple([v*self.scale for v in rawVals])

class GyroscopeSensor(SensorBase):
    svcUUID = _TI_UUID(0xAA50)
    dataUUID = _TI_UUID(0xAA51)
    ctrlUUID = _TI_UUID(0xAA52)
    sensorOn = struct.pack("B", 0x07)

    def __init__(self, periph):
        SensorBase.__init__(self, periph)

    def read(self):
        '''Returns (x,y,z) rate in deg/sec'''
        x_y_z = struct.unpack('<hhh', self.data.read())
        return tuple([250.0 * (v/32768.0) for v in x_y_z])


class GyroscopeSensorMPU9250:
    def __init__(self, sensor_):
        self.sensor = sensor_
        self.scale = 500.0/65536.0

    def enable(self):
        self.sensor.enable(self.sensor.GYRO_XYZ)

    def disable(self):
        self.sensor.disable(self.sensor.GYRO_XYZ)

    def read(self):
        '''Returns (x_gyro, y_gyro, z_gyro) in units of degrees/sec'''
        rawVals = self.sensor.rawRead()[0:3]
        return tuple([v*self.scale for v in rawVals])


class BatterySensor(SensorBase):
    svcUUID = UUID("0000180f-0000-1000-8000-00805f9b34fb")
    dataUUID = UUID("00002a19-0000-1000-8000-00805f9b34fb")
    ctrlUUID = None
    sensorOn = None

    def __init__(self, periph):
        SensorBase.__init__(self, periph)

    def read(self):
        '''Returns the battery level in percent'''
        val = ord(self.data.read())
        return val

class LEDAndBuzzer(SensorBase):
    '''
    UUID obtained from:
    https://usermanual.wiki/Document/CC265020SensorTag20Users20Guide2020Texas20Instruments20Wiki.2070227354.pdf
    (Under IO service)
    ''' 
    svcUUID = _TI_UUID(0xAA64)
    dataUUID = _TI_UUID(0xAA65)
    ctrlUUID = _TI_UUID(0xAA66)
    sensorOn = struct.pack("B", 0x01)

    def __init__(self, periph):
        SensorBase.__init__(self, periph)

    def activate_buzzer(self):
        print("Buzzer activated")
        self.data.write(struct.pack("B", 0x04), withResponse=False) # TODO: change value from 0x1 (red LED) to 0x4 (buzzer)
    def deactivate_buzzer(self):
        print("Buzzer deactivated")
        self.data.write(struct.pack("B", 0x00), withResponse=False)

class SensorTag(Peripheral):
    def __init__(self, addr, version=AUTODETECT):
        Peripheral.__init__(self, addr)
        if version == AUTODETECT:
            svcs = self.discoverServices()
            if _TI_UUID(0xAA70) in svcs:
                version = SENSORTAG_2650
            else:
                version = SENSORTAG_V1

        fwVers = self.getCharacteristics(
            uuid=AssignedNumbers.firmwareRevisionString)
        if len(fwVers) >= 1:
            self.firmwareVersion = fwVers[0].read().decode("utf-8")
        else:
            self.firmwareVersion = u''

        self._mpu9250 = MovementSensorMPU9250(self)
        self.accelerometer = AccelerometerSensorMPU9250(self._mpu9250)
        self.gyroscope = GyroscopeSensorMPU9250(self._mpu9250)
        self.battery = BatterySensor(self)
        self.IO = LEDAndBuzzer(self)

def notify_edge_devices(tag, host):
    global consecutive_bad_postures
    global current_posture
    global buzzer_is_active
    if current_posture is not None:
        if current_posture == "good":
            consecutive_bad_postures = 0
            if buzzer_is_active:
                tag.IO.deactivate_buzzer()
                buzzer_is_active = False
        else:
            consecutive_bad_postures += 1 
        current_posture = None

    if consecutive_bad_postures >= 5 and not buzzer_is_active:
        tag.IO.activate_buzzer()
        buzzer_is_active = True

def main():
    import time
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('host', action='store', help='MAC of BT device')
    parser.add_argument('-n', action='store', dest='count', default=0,
                        type=int, help="Number of times to loop data")
    parser.add_argument('-t', action='store', type=float,
                        default=5.0, help='time between polling')

    arg = parser.parse_args(sys.argv[1:])

    print('Connecting to ' + arg.host)
    global tag
    tag = SensorTag(arg.host)

    # Enabling the relevant sensors
    tag.accelerometer.enable()
    tag.gyroscope.enable()
    tag.battery.enable()
    tag.IO.enable()

    # Some sensors (e.g., temperature, accelerometer) need some time for initialization.
    # Not waiting here after enabling a sensor, the first read value might be empty or incorrect.
    time.sleep(5.0)

    tag.IO.deactivate_buzzer()

    signal.signal(signal.SIGINT, sigint_handler)
    signal.signal(signal.SIGTERM, sigterm_handler)

    # csv_reader = CsvReader(arg.host)
    mosquitto = MQTT()
    server = mosquitto.setup("54.255.139.76")
    # server = mosquitto.setup("127.0.0.1")

    battery = tag.battery.read()
    print("Battery: ", battery)
    mosquitto.send_battery_data(server, arg.host, battery)

    counter = 1
    rows = []
    while True:
        data = []

        acc = tag.accelerometer.read()
        data += list(acc)
        print("Accelerometer: ", acc)

        # gyro = tag.gyroscope.read()
        data += list((0,0,0))
        # print("Gyroscope: ", gyro)

        curr_battery = tag.battery.read()
        if curr_battery != battery:
            battery = curr_battery
            mosquitto.send_battery_data(server, arg.host, battery)
            print("Battery: ", battery)

        data += [posture_label]
        # uncomment to collect data
        # csv_reader.write_row(data) 
        rows.append(data)
        counter += 1

        if len(rows) == 10:
            mosquitto.send_data(server, arg.host, counter // 10, list(rows))
            rows = rows[2:]
            
        # Send task to buffer in the while loop to avoid race condition
        # notify_edge_devices(tag, arg.host)

        tag.waitForNotifications(arg.t)
    tag.disconnect()
    del tag


if __name__ == "__main__":
    main()