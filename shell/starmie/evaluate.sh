echo "Evaluation using starmie model"
python scripts/starmie/evaluate_benchmark.py santos --searcher_type bounds
python scripts/starmie/evaluate_benchmark.py santos --ao shuffle_col --searcher_type bounds

python scripts/starmie/evaluate_benchmark.py santos --searcher_type lsh
python scripts/starmie/evaluate_benchmark.py santos --ao shuffle_col --searcher_type lsh

python scripts/starmie/evaluate_benchmark.py santos --searcher_type hnsw
python scripts/starmie/evaluate_benchmark.py santos --ao shuffle_col --searcher_type hnsw

python scripts/starmie/evaluate_benchmark.py santos --searcher_type cluster
python scripts/starmie/evaluate_benchmark.py santos --ao shuffle_col --searcher_type cluster

python scripts/starmie/evaluate_benchmark.py santos --searcher_type faiss
python scripts/starmie/evaluate_benchmark.py santos --ao shuffle_col --searcher_type faiss

python scripts/starmie/evaluate_benchmark.py santos --searcher_type faiss --pooling max
python scripts/starmie/evaluate_benchmark.py santos --ao shuffle_col --searcher_type faiss --pooling max

python scripts/starmie/evaluate_benchmark.py santos --searcher_type faiss --pooling mean
python scripts/starmie/evaluate_benchmark.py santos --ao shuffle_col --searcher_type faiss --pooling mean



python scripts/starmie/evaluate_benchmark.py tus --searcher_type bounds
python scripts/starmie/evaluate_benchmark.py tus --ao shuffle_col --searcher_type bounds

python scripts/starmie/evaluate_benchmark.py tus --searcher_type lsh
python scripts/starmie/evaluate_benchmark.py tus --ao shuffle_col --searcher_type lsh

python scripts/starmie/evaluate_benchmark.py tus --searcher_type hnsw
python scripts/starmie/evaluate_benchmark.py tus --ao shuffle_col --searcher_type hnsw

python scripts/starmie/evaluate_benchmark.py tus --searcher_type cluster
python scripts/starmie/evaluate_benchmark.py tus --ao shuffle_col --searcher_type cluster

python scripts/starmie/evaluate_benchmark.py tus --searcher_type faiss
python scripts/starmie/evaluate_benchmark.py tus --ao shuffle_col --searcher_type faiss

python scripts/starmie/evaluate_benchmark.py tus --searcher_type faiss --pooling max
python scripts/starmie/evaluate_benchmark.py tus --ao shuffle_col --searcher_type faiss --pooling max

python scripts/starmie/evaluate_benchmark.py tus --searcher_type faiss --pooling mean
python scripts/starmie/evaluate_benchmark.py tus --ao shuffle_col --searcher_type faiss --pooling mean



python scripts/starmie/evaluate_benchmark.py tusLarge --searcher_type bounds
python scripts/starmie/evaluate_benchmark.py tusLarge --ao shuffle_col --searcher_type bounds

python scripts/starmie/evaluate_benchmark.py tusLarge --searcher_type lsh
python scripts/starmie/evaluate_benchmark.py tusLarge --ao shuffle_col --searcher_type lsh

python scripts/starmie/evaluate_benchmark.py tusLarge --searcher_type hnsw
python scripts/starmie/evaluate_benchmark.py tusLarge --ao shuffle_col --searcher_type hnsw

python scripts/starmie/evaluate_benchmark.py tusLarge --searcher_type cluster
python scripts/starmie/evaluate_benchmark.py tusLarge --ao shuffle_col --searcher_type cluster

python scripts/starmie/evaluate_benchmark.py tusLarge --searcher_type faiss
python scripts/starmie/evaluate_benchmark.py tusLarge --ao shuffle_col --searcher_type faiss

python scripts/starmie/evaluate_benchmark.py tusLarge --searcher_type faiss --pooling max
python scripts/starmie/evaluate_benchmark.py tusLarge --ao shuffle_col --searcher_type faiss --pooling max

python scripts/starmie/evaluate_benchmark.py tusLarge --searcher_type faiss --pooling mean
python scripts/starmie/evaluate_benchmark.py tusLarge --ao shuffle_col --searcher_type faiss --pooling mean

