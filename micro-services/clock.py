import argparse
import sys
import requests
import time

ARG_PARSER = argparse.ArgumentParser(description='Notify other micro-services like a heartbeat ...')
ARG_PARSER.add_argument('--interval', dest='interval', type=float, required=True, help='signal interval in seconds')
ARG_PARSER.add_argument('--begin-timestamp', dest='begin_timestamp', type=int, default=int(time.time()), help='The begining timestamp that will be incremented as much as --real-interval seconds at each signal')
ARG_PARSER.add_argument('--real-interval', dest='real_interval', type=int, required=True, help='The number of seconds being added to the heartbeat signal current_timestamp parameter')
ARG_PARSER.add_argument('--endpoints', dest='endpoints', type=str, required=True, nargs='*', help='The endpoints to send signals to')
ARG_PARSER.add_argument('--request-timeout', dest='request_timeout', type=int, help='Notification request timeout', default=5)

ARGS = ARG_PARSER.parse_args()

print("Passed arguments: ")
print(ARGS)

def main():
    current_timestamp = ARGS.begin_timestamp
    while True:
        for e in ARGS.endpoints:
            try:
                response = requests.get(e, verify=False, timeout=ARGS.request_timeout, params={'current_timestamp': current_timestamp})
            except Exception as err:
                print(err)
        current_timestamp = int(current_timestamp + ARGS.real_interval)
           
        time.sleep(ARGS.interval)


if __name__ == "__main__":
    main()
