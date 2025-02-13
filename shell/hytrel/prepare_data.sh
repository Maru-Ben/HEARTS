#!/bin/bash

benchmarks=(
    "santos"
    "santos-p-col"
    # "pylon"
    # "pylon-p-col"
    # "tus"
    # "tus-p-col"
    # "tusLarge"
    # "tusLarge-p-col"
)

for benchmark in "${benchmarks[@]}"; do
    echo "Processing $benchmark..."
    echo "Step 1: Converting CSV to JSONL..."
    python scripts/hytrel/csv_to_jsonl.py "$benchmark"
    
    echo "Step 2: Running parallel clean..."
    python scripts/hytrel/parallel_clean.py "$benchmark"
    
    echo "Completed $benchmark"
    echo "-------------------"
done

echo "All benchmarks processed!"