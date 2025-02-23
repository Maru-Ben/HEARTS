###### Make sure to do these steps first before running setup.sh
# conda env create -f environment.yml
# conda activate hearts

# 1) install PyTorch first
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# 2) Install PyG, torch-scatter, torch-sparse, and torch-geometric.
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.3.0+cu121.html

# 3) Install the rest of your Python packages
pip install -r requirements.txt

# 4) Download SpaCy model
python -m spacy download en_core_web_sm