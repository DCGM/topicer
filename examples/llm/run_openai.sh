#!/bin/bash
# RUN THIS SCRIPT FROM THE ROOT DIRECTORY OF THE REPOSITORY
# add the root directory to PYTHONPATH
export PYTHONPATH=$(pwd):$PYTHONPATH
./examples/llm/run_call_openai.py run