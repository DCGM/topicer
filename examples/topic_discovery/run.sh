#!/bin/bash
# RUN THIS SCRIPT FROM THE ROOT DIRECTORY OF THE REPOSITORY
# add the root directory to PYTHONPATH
export PYTHONPATH=$(pwd):$PYTHONPATH
./examples/topic_discovery/run_topic_discovery.py run