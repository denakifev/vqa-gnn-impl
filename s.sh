mkdir -p data/raw/vcr data/vcr/visual_features data/vcr/knowledge_graphs
find data/raw/vcr data/vcr data/conceptnet -maxdepth 3 \( -type f -o -type l -o -type d \) -print
