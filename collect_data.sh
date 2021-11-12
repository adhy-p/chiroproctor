#!/bin/bash

function sigint_handler() {
    echo "SIGINT received"
    # kill -2 ${PID1} ${PID2}
}

function sigquit_handler() {
    kill -15 ${PID1} ${PID2}
    exit 0
}

trap 'sigint_handler' 2
trap 'sigquit_handler' 3

is_running="true"
python3 main.py -t 0.2 -A -G 54:6C:0E:52:EF:9E &
PID1=$!
echo $PID1
python3 main.py -t 0.2 -A -G 54:6C:0E:B6:D3:85 &
PID2=$!
echo $PID2

while [ $is_running=="true" ]; do
    sleep 0.0001
done
