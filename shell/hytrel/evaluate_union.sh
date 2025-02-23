python scripts/hytrel/evaluate_benchmark_union.py santos  --searcher_type cluster
python scripts/hytrel/evaluate_benchmark_union.py santos  --searcher_type bounds
python scripts/hytrel/evaluate_benchmark_union.py santos  --searcher_type lsh
python scripts/hytrel/evaluate_benchmark_union.py santos  --searcher_type hnsw
python scripts/hytrel/evaluate_benchmark_union.py santos  --searcher_type faiss --pooling max
python scripts/hytrel/evaluate_benchmark_union.py santos  --searcher_type faiss --pooling mean
python scripts/hytrel/evaluate_benchmark_union.py santos  --searcher_type faiss --pooling None

python scripts/hytrel/evaluate_benchmark_union.py tus  --searcher_type bounds
python scripts/hytrel/evaluate_benchmark_union.py tus  --searcher_type lsh
python scripts/hytrel/evaluate_benchmark_union.py tus  --searcher_type hnsw
python scripts/hytrel/evaluate_benchmark_union.py tus  --searcher_type cluster
python scripts/hytrel/evaluate_benchmark_union.py tus  --searcher_type faiss --pooling max
python scripts/hytrel/evaluate_benchmark_union.py tus  --searcher_type faiss --pooling mean
python scripts/hytrel/evaluate_benchmark_union.py tus  --searcher_type faiss --pooling None

python scripts/hytrel/evaluate_benchmark_union.py tusLarge  --searcher_type bounds
python scripts/hytrel/evaluate_benchmark_union.py tusLarge  --searcher_type lsh
python scripts/hytrel/evaluate_benchmark_union.py tusLarge  --searcher_type hnsw
python scripts/hytrel/evaluate_benchmark_union.py tusLarge  --searcher_type cluster
python scripts/hytrel/evaluate_benchmark_union.py tusLarge  --searcher_type faiss --pooling max
python scripts/hytrel/evaluate_benchmark_union.py tusLarge  --searcher_type faiss --pooling mean
python scripts/hytrel/evaluate_benchmark_union.py tusLarge  --searcher_type faiss --pooling None
