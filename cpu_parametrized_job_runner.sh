#!/bin/bash

source ~/.profile

source activate tensorflow_cpu
python "$@"
