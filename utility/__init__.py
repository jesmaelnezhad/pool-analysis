import logging

# Initialize the logging utility
FORMAT = '%(asctime)s - %(name)s -  %(levelname)s - %(message)s'
logging.basicConfig(level=logging.DEBUG, format=FORMAT)


def logger(name):
    logger_ = logging.getLogger(name)
    logger_.setLevel(logging.DEBUG)
    return logger_
