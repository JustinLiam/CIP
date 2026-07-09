#!/bin/bash

cd "$(dirname "$0")/.."

echo "Fetching Delphi data..."
python3 scripts/delphi_api.py

echo "Fetching Census data..."
python3 scripts/census.py

echo "Generating placeholder networks..."
cd networks
python3 initialize_experiment.py <<INPUT
yes
test
1
no
INPUT
cd ..

echo "Generating county folders..."
python3 scripts/process_counties.py

echo "Data preparation complete!"
