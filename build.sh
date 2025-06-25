#!/bin/bash
set -e

# Update pip and install ddependancies
pip install --upgrade pip
pip install -r requirements.txt

# Download portable Chrome for Kaleido
plotly_get_chrome