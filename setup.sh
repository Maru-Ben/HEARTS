# 1) Source the conda.sh file to set up conda in this shell session.
source "$(conda info --base)/etc/profile.d/conda.sh"

# 2) Create the environment from environment.yml
conda env create -f environment.yml

# 3) Activate
conda activate hearts

# 4) Now install PyTorch first
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# 5) Install PyG, torch-scatter, torch-sparse, and torch-geometric.
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.3.0+cu121.html

# 6) Install the rest of your Python packages
pip install -r requirements.txt

# 7) Download SpaCy model
python -m spacy download en_core_web_sm