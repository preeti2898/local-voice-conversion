#!/bin/bash
cd "$(dirname "$0")" || exit 1
export HF_HUB_OFFLINE=1
export GRADIO_ANALYTICS_ENABLED=False
python local_app.py
