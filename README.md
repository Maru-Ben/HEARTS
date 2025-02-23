# HEARTS: Hypergraph-based Related Table Search

# Setup Instructions

## 1. Environment Setup

Set up the environment by running the following commands in your terminal:

```bash
# Create the conda environment using the provided YAML file.
conda env create -f environment.yml

# Activate the newly created environment.
conda activate hearts

# Run the initial setup script.
bash setup.sh
```

These commands install all required dependencies and configure your environment for the project.

---

## 2. Download Benchmarks & Pretrained Checkpoints

You can download all necessary checkpoints and benchmarks **automatically** using `download.sh` **or manually** from the links below.

### Automatic Download (Recommended)

Simply run:

```bash
bash download.sh
```

This script downloads and extracts all resources into their proper directories.

### Manual Download (If Needed)

If you prefer to download files individually, use the table below to access each resource directly. You will need to extract them manually into the correct directories.

#### Checkpoints

| Component   | Description                   | Download Link                                                                                                                         |
|-------------|-------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| **HyTrel**  | Pretrained HyTrel checkpoint  | [Download HyTrel](https://nuage.lip6.fr/index.php/s/LW6qQZ4jeNkBNSW/download/hytrel.tar.gz)                                          |
| **Fasttext**| Pretrained Fasttext model     | [Download Fasttext](https://nuage.lip6.fr/index.php/s/KYYXfGncwiFSKd7/download/fasttext.tar.gz)                                        |

#### Benchmarks

| Benchmark    | Description                  | Download Link                                                                                                                         |
|--------------|------------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| **Santos**   | Santos Benchmark             | [Download Santos](https://nuage.lip6.fr/index.php/s/dXZ9fbtXfsptHoZ/download/santos.tar.gz)                                            |
| **TUS**      | TUS Benchmark                | [Download TUS](https://nuage.lip6.fr/index.php/s/Np5CLbENHWwHrzF/download/tus.tar.gz)                                                  |
| **TUS Large**| TUS Large Benchmark          | [Download TUS Large](https://nuage.lip6.fr/index.php/s/cJJwtdzW6Nt6ssb/download/tusLarge.tar.gz)                                        |
| **Wiki-Join**| Wiki-Join Benchmark          | [Download Wiki-Join](https://nuage.lip6.fr/index.php/s/LKWeDCZ9MQcTMCN/download/wiki-join.tar.gz)                                        |

If downloading manually, ensure you extract the files into the correct locations.

---

## 3. Create Column-Shuffled Versions for Table Union Search

To evaluate HEARTS under adversarial conditions—i.e., with columns shuffled randomly—you **must** run the following script **after downloading the datasets**:

```bash
bash prepare_data.sh
```

This script will:
- Generate a shuffled version of each benchmark (stored in directories like `datalake-p-col/`).
- Create truncated versions for use with HyTrel, saving them in `datalake_hytrel/` and `datalake_hytrel_p-col/`.

This preprocessing is essential because HEARTS requires truncated tables to avoid memory issues due to massive hypergraph representations.



# 4) Run the evaluation scripts 
## HyTrel

## Starmie

## DeepJoin (notebook)

# 5) Visualize the results
