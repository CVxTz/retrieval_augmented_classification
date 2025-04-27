# https://github.com/iliaschalkidis/lmtc-eurlex57k
# Define directories
base_dir="data"
datasets_dir="$base_dir/datasets"
eurlex57k_dir="$datasets_dir/EURLEX57K"

# Create directories
mkdir -p "$base_dir"
mkdir -p "$datasets_dir"
mkdir -p "$eurlex57k_dir"

# Download and extract the dataset
wget -O data/datasets/datasets.zip http://nlp.cs.aueb.gr/software_and_datasets/EURLEX57K/datasets.zip
unzip data/datasets/datasets.zip -d data/datasets/EURLEX57K
rm data/datasets/datasets.zip
rm -rf data/datasets/EURLEX57K/__MACOSX
mv data/datasets/EURLEX57K/dataset/* data/datasets/EURLEX57K/
rm -rf data/datasets/EURLEX57K/dataset
wget -O data/datasets/EURLEX57K/EURLEX57K.json http://nlp.cs.aueb.gr/software_and_datasets/EURLEX57K/eurovoc_en.json
