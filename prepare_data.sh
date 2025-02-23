echo "Preparing data for HEARTS (HyTrel requires truncation)"
# python data/shuffle.py -i data/santos/datalake/ -o data/santos/datalake_hytrel/ -r 30 -c 20 -t -C -s sort_by_tfidf_improved
# python data/shuffle.py -i data/santos/datalake_hytrel/ -o data/santos/datalake_hytrel_p-col/ -r -1 -c -1 -t -C 

python data/shuffle.py -i data/tus/datalake/ -o data/tus/datalake_hytrel/ -r 30 -c 20 -t -C -s sort_by_tfidf_improved
python data/shuffle.py -i data/tus/datalake_hytrel/ -o data/tus/datalake_hytrel_p-col/ -r -1 -c -1 -t -C 

python data/shuffle.py -i data/tusLarge/datalake/ -o data/tusLarge/datalake_hytrel/ -r 30 -c 20 -t -C -s sort_by_tfidf_improved
python data/shuffle.py -i data/tusLarge/datalake_hytrel/ -o data/tusLarge/datalake_hytrel_p-col/ -r -1 -c -1 -t -C 




echo "Preparing data for Starmie (doesn't requires truncation)"
python data/shuffle.py -i data/santos/datalake/ -o data/santos/datalake-p-col/ -r -1 -c -1 -t -C
python data/shuffle.py -i data/tus/datalake/ -o data/tus/datalake-p-col/ -r -1 -c -1 -t -C
python data/shuffle.py -i data/tusLarge/datalake/ -o data/tusLarge/datalake-p-col/ -r -1 -c -1 -t -C