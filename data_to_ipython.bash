#!/bin/bash

filename=$(basename "$1")

ipython -i --matplotlib=qt4 load_data.py -- -f $1

