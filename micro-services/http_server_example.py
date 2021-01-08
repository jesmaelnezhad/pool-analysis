# Base from: https://pythonbasics.org/webserver/
# Python 3 server example
from http.server import BaseHTTPRequestHandler, HTTPServer
import sys
from urllib.parse import urlparse
import argparse

ARG_PARSER = argparse.ArgumentParser(description='Notify other micro-services like a heartbeat ...')
ARG_PARSER.add_argument('--hostname', dest='hostname', type=str, required=True, help='Hostname to bind')
ARG_PARSER.add_argument('--port', dest='port', type=int, required=True, help='Port to bind')

ARGS = ARG_PARSER.parse_args()

print("Passed arguments: ")
print(ARGS)


class MyServer(BaseHTTPRequestHandler):
    def do_GET(self):
        query = urlparse(self.path).query
        request_parameters = dict(qc.split("=") for qc in query.split("&"))

        print("Current timestamp: " + request_parameters['current_timestamp'])
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()

if __name__ == "__main__":
    webServer = HTTPServer((ARGS.hostname, ARGS.port), MyServer)
    print("Server started http://%s:%s" % (ARGS.hostname, ARGS.port))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")
