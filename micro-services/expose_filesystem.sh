#!/bin/bash
set -x
python3 -m http.server $1 --bind $2
