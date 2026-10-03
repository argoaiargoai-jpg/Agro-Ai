#!/bin/sh
# Zips the training code so you can upload it to Google Colab (or Google Drive).
cd "$(dirname "$0")" && rm -f agroai_ml_bundle.zip && zip -q agroai_ml_bundle.zip common.py prepare_data.py mldata.py train.py decision.py evaluate.py export_onnx.py requirements-train.txt && echo "created ml/agroai_ml_bundle.zip"
