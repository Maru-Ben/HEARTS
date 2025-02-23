echo "Default union evaluation: HEARTS / SANTOS, TUS, TUS-LARGE"
./shell/hytrel/extract_vectors_union.sh
./shell/hytrel/evaluate_union.sh

echo "Default join evaluation: HEARTS / Wiki-Join"
./shell/hytrel/extract_vectors_join.sh
./shell/hytrel/evaluate_join.sh