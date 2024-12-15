# #!/bin/bash

# echo "Evaluation using pretrained model"
# python scripts/hytrel/evaluate_benchmark.py santos --model_type pretrained --searcher_type bounds
# python scripts/hytrel/evaluate_benchmark.py santos --model_type pretrained --searcher_type lsh
# python scripts/hytrel/evaluate_benchmark.py santos --model_type pretrained --searcher_type hnsw

# python scripts/hytrel/evaluate_benchmark.py pylon --model_type pretrained --searcher_type bounds
# python scripts/hytrel/evaluate_benchmark.py pylon --model_type pretrained --searcher_type lsh
# python scripts/hytrel/evaluate_benchmark.py pylon --model_type pretrained --searcher_type hnsw

# python scripts/hytrel/evaluate_benchmark.py tus --model_type pretrained --searcher_type bounds
# python scripts/hytrel/evaluate_benchmark.py tus --model_type pretrained --searcher_type lsh
# python scripts/hytrel/evaluate_benchmark.py tus --model_type pretrained --searcher_type hnsw

# python scripts/hytrel/evaluate_benchmark.py tusLarge --model_type pretrained --searcher_type bounds
# python scripts/hytrel/evaluate_benchmark.py tusLarge --model_type pretrained --searcher_type lsh
# python scripts/hytrel/evaluate_benchmark.py tusLarge --model_type pretrained --searcher_type hnsw


echo "Evaluation using scratch model"
python scripts/hytrel/evaluate_benchmark.py santos --model_type scratch --searcher_type bounds
python scripts/hytrel/evaluate_benchmark.py santos --model_type scratch --searcher_type lsh
python scripts/hytrel/evaluate_benchmark.py santos --model_type scratch --searcher_type hnsw

# python scripts/hytrel/evaluate_benchmark.py tus --model_type scratch --searcher_type bounds
# python scripts/hytrel/evaluate_benchmark.py tus --model_type scratch --searcher_type lsh
# python scripts/hytrel/evaluate_benchmark.py tus --model_type scratch --searcher_type hnsw

# python scripts/hytrel/evaluate_benchmark.py tusLarge --model_type scratch --searcher_type bounds
# python scripts/hytrel/evaluate_benchmark.py tusLarge --model_type scratch --searcher_type lsh
# python scripts/hytrel/evaluate_benchmark.py tusLarge --model_type scratch --searcher_type hnsw

# python scripts/hytrel/evaluate_benchmark.py pylon --model_type scratch --searcher_type bounds
# python scripts/hytrel/evaluate_benchmark.py pylon --model_type scratch --searcher_type lsh
# python scripts/hytrel/evaluate_benchmark.py pylon --model_type scratch --searcher_type hnsw


# echo "Evaluation using finetuned model"
# python scripts/hytrel/evaluate_benchmark.py santos --model_type finetuned --searcher_type bounds
# python scripts/hytrel/evaluate_benchmark.py santos --model_type finetuned --searcher_type lsh
# python scripts/hytrel/evaluate_benchmark.py santos --model_type finetuned --searcher_type hnsw

# python scripts/hytrel/evaluate_benchmark.py tus --model_type finetuned --searcher_type bounds
# python scripts/hytrel/evaluate_benchmark.py tus --model_type finetuned --searcher_type lsh
# python scripts/hytrel/evaluate_benchmark.py tus --model_type finetuned --searcher_type hnsw

# python scripts/hytrel/evaluate_benchmark.py tusLarge --model_type finetuned --searcher_type bounds
# python scripts/hytrel/evaluate_benchmark.py tusLarge --model_type finetuned --searcher_type lsh
# python scripts/hytrel/evaluate_benchmark.py tusLarge --model_type finetuned --searcher_type hnsw

# python scripts/hytrel/evaluate_benchmark.py pylon --model_type finetuned --searcher_type bounds
# python scripts/hytrel/evaluate_benchmark.py pylon --model_type finetuned --searcher_type lsh
# python scripts/hytrel/evaluate_benchmark.py pylon --model_type finetuned --searcher_type hnsw

