# 1) Setup the environment
Step 1:
conda env create -f environment.yml

Step 2:
conda activate hearts

Step 3:
bash setup.sh

# 2) Download the benchmarks & pretrained checkpoints
You can download the benchmarks used for the evaluation in the poster, as well as the hytrel checkpoint trained with the contrastive loss, and the fasttext pretrained model, directly using download.sh, you can also download each one of them individually through this table : 

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



However, you'll have to place them in the right directory yourself and extract them. download.sh already does that for you, all you gotta do is "bash download.sh"


# 3) create column-shuffled versions of benchmarks for Table Union Search

In order to show the effectiveness of HEARTS compared to LM-based methods like STARMIE, we evaluate the adversarial scenario where columns of different data lakes are shuffled randomly. To this end, you can run prepare_data.sh ia bash prepare_data.sh which will automatically created a shuffled version of each benchmark by constructing alongside the "datalake/" folder of each benchmark, a "datalake-p-col/" folder where each csv has a shuffled column order randomly. Please keep in mind that since HEARTS uses HyTrel, and since HyTrel requires the tables to be truncated, we create truncated versions of each benchmark and shuffle the truncated version in order to create datalake_hytrel and datalake_hytrel_p-col folders for each benchmark. This is part of the method's preprocessing. STARMIE can handle bigger datasets because sampling is done later on, HEARTS needs the tables to be processed truncated to not result in massive hypergraph representations that could saturate the memory. 

# 4) Run the evaluation scripts 
## HyTrel

## Starmie

## DeepJoin (notebook)

# 5) Visualize the results
