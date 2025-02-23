conda create -n testme python=3.11 -y
conda activate testme
pip install pip==24.0

pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.3.0+cu121.html

pip install -r requirements.txt

python -m spacy download en_core_web_sm

# If you get the `GLIBCXX_3.4.21' not found error, run the following commands:
conda install libgcc -y
conda install conda-forge::libstdcxx-ng
