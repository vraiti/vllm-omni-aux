#!/bin/bash
source $HOME/.bashrc
cloudexe --queue --gpuspec EUN1H100x1 -- /usr/local/bin/python bash.py $@
