echo "Downloading and extracting the checkpoints"
mkdir -p checkpoints

echo "1/2 Downloading and extracting HyTrel"
wget -O checkpoints/hytrel.tar.gz https://nuage.lip6.fr/index.php/s/LW6qQZ4jeNkBNSW/download/hytrel.tar.gz
tar -xzvf checkpoints/hytrel.tar.gz -C checkpoints
rm checkpoints/hytrel.tar.gz

echo "2/2 Downloading and extracting Fasttext"
wget -O checkpoints/fasttext.tar.gz https://nuage.lip6.fr/index.php/s/KYYXfGncwiFSKd7/download/fasttext.tar.gz
tar -xzvf checkpoints/fasttext.tar.gz -C checkpoints
rm checkpoints/fasttext.tar.gz


echo "Downloading and extracting the benchmarks"
mkdir -p data

echo "1/4 Downloading and extracting Santos"
wget -O data/santos.tar.gz https://nuage.lip6.fr/index.php/s/dXZ9fbtXfsptHoZ/download/santos.tar.gz
tar -xzvf data/santos.tar.gz -C data
rm data/santos.tar.gz

echo "2/4 Downloading and extracting TUS"
wget -O data/tus.tar.gz https://nuage.lip6.fr/index.php/s/Np5CLbENHWwHrzF/download/tus.tar.gz
tar -xzvf data/tus.tar.gz -C data
rm data/tus.tar.gz

echo "3/4 Downloading and extracting TUS Large"
wget -O data/tusLarge.tar.gz https://nuage.lip6.fr/index.php/s/cJJwtdzW6Nt6ssb/download/tusLarge.tar.gz
tar -xzvf data/tusLarge.tar.gz -C data
rm data/tusLarge.tar.gz

echo "4/4 Downloading and extracting Wiki-Join"
wget -O data/wiki-join.tar.gz https://nuage.lip6.fr/index.php/s/LKWeDCZ9MQcTMCN/download/wiki-join.tar.gz
tar -xzvf data/wiki-join.tar.gz -C data
rm data/wiki-join.tar.gz



