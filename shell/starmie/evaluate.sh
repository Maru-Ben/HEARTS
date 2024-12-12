# #!/bin/bash

echo "Evaluation using starmie model"
# python scripts/starmie/evaluate_benchmark.py santos --searcher_type bounds
python scripts/starmie/evaluate_benchmark.py santos --searcher_type lsh
python scripts/starmie/evaluate_benchmark.py santos --searcher_type hnsw

# python scripts/starmie/evaluate_benchmark.py pylon --searcher_type bounds
python scripts/starmie/evaluate_benchmark.py pylon --searcher_type lsh
python scripts/starmie/evaluate_benchmark.py pylon --searcher_type hnsw

# python scripts/starmie/evaluate_benchmark.py tus --searcher_type bounds
python scripts/starmie/evaluate_benchmark.py tus --searcher_type lsh
python scripts/starmie/evaluate_benchmark.py tus --searcher_type hnsw

# python scripts/starmie/evaluate_benchmark.py tusLarge --searcher_type bounds
python scripts/starmie/evaluate_benchmark.py tusLarge --searcher_type lsh
python scripts/starmie/evaluate_benchmark.py tusLarge --searcher_type hnsw

