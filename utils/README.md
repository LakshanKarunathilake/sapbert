# SAPBERT FAISS Utility

A comprehensive utility for generating SAPBERT embeddings from Wikidata CSV files and creating efficient FAISS indexes for semantic similarity search. The utility is split into two optimized components: index creation and search functionality.

## Features

- **SAPBERT Integration**: Uses state-of-the-art biomedical entity embeddings
- **Multiple Index Types**: Support for exact (Flat), approximate (IVF), and fast (HNSW) search
- **Data Preprocessing**: Automatic cleaning and filtering of Wikidata aliases
- **Modular Design**: Separate optimized tools for index creation and searching
- **Command Line Interface**: Easy-to-use CLI with sensible defaults
- **GPU Support**: Automatic GPU detection and usage when available
- **Export Options**: JSON export for search results

## Installation

```bash
# Install required packages
pip install torch transformers faiss-cpu pandas numpy tqdm

# For GPU support (optional but recommended)
pip install faiss-gpu

# Download the utility files
# - create_sapbert_index.py (index creation)
# - search_sapbert_index.py (search and query)
# - README.md (this file)
```

## Quick Start

### 1. Prepare Your CSV File

Your CSV file must have these columns:
- `qid`: Wikidata QID (e.g., "Q12345")
- `aliases`: Pipe-separated aliases (e.g., "diabetes||diabetes mellitus||DM")

Example CSV:
```csv
qid,aliases
Q8071,machine learning||ML||statistical learning
Q12206,diabetes||diabetes mellitus||DM||diabetes type 2
Q2539,elephant||elephants||Elephantidae
```

### 2. Create Index

```bash
# Basic usage with defaults
python create_sapbert_index.py

# Custom parameters
python create_sapbert_index.py \
    --csv_path my_data.csv \
    --output_dir ./my_indexes \
    --index_name biomedical_entities \
    --index_type Flat \
    --batch_size 32
```

### 3. Search the Index

```bash
# Basic search
python search_sapbert_index.py --query "heart disease"

# Custom search
python search_sapbert_index.py \
    --index_path ./my_indexes/biomedical_entities \
    --query "diabetes mellitus" \
    --k 5
```

## File Structure

```
your_project/
├── create_sapbert_index.py    # Index creation utility
├── search_sapbert_index.py    # Search and query utility
└── README.md                  # This documentation
```

When you create an index, the following files are generated:

```
indexes/
├── wikidata_index.faiss              # FAISS index file
├── wikidata_index_metadata.pkl       # Entity metadata
├── wikidata_index_config.json        # Index configuration
└── wikidata_index_processed_data.csv # Processed data (reference)
```

## Index Creation (`create_sapbert_index.py`)

### Command Line Options

```bash
python create_sapbert_index.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--csv_path` | `data/wikidata.csv` | Path to CSV file with qid and aliases columns |
| `--output_dir` | `./indexes` | Directory to save index files |
| `--index_name` | `wikidata_index` | Name for index files |
| `--index_type` | `IVF` | FAISS index type (Flat/IVF/HNSW) |
| `--batch_size` | `16` | Batch size for embedding generation |
| `--model_name` | `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` | SAPBERT model |
| `--validate_only` | `False` | Only validate CSV without creating index |
| `--verbose` | `False` | Enable verbose logging |

### Examples

```bash
# Create high-accuracy index for small dataset
python create_sapbert_index.py \
    --csv_path small_entities.csv \
    --index_type Flat

# Create fast index for large dataset
python create_sapbert_index.py \
    --csv_path large_entities.csv \
    --index_type HNSW \
    --batch_size 64

# Validate CSV format only
python create_sapbert_index.py \
    --csv_path my_data.csv \
    --validate_only

# Use different SAPBERT model
python create_sapbert_index.py \
    --model_name "dmis-lab/biobert-base-cased-v1.1"
```

## Search and Query (`search_sapbert_index.py`)

### Search Modes

The search utility supports multiple operation modes:

| Mode | Description |
|------|-------------|
| `search` | Single text search (default) |
| `batch_search` | Multiple text searches |
| `get_entity` | Look up entity by QID |
| `similar` | Find entities similar to a given QID |
| `stats` | Show index statistics |

### Command Line Options

```bash
python search_sapbert_index.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--mode` | `search` | Search mode |
| `--index_path` | `./indexes/wikidata_index` | Path to index files (without extension) |
| `--query` | `diabetes mellitus` | Search query text |
| `--queries` | `diabetes,cancer,heart disease` | Comma-separated queries for batch search |
| `--qid` | `Q12206` | QID for entity lookup or similarity search |
| `--k` | `10` | Number of results to return |
| `--export` | `None` | Export results to JSON file |
| `--verbose` | `False` | Enable verbose logging |

### Search Examples

```bash
# Single search
python search_sapbert_index.py --query "machine learning" --k 5

# Batch search
python search_sapbert_index.py \
    --mode batch_search \
    --queries "diabetes,cancer,pneumonia,asthma" \
    --k 3

# Get index statistics
python search_sapbert_index.py --mode stats

# Look up specific entity
python search_sapbert_index.py --mode get_entity --qid Q12206

# Find similar entities to a QID
python search_sapbert_index.py --mode similar --qid Q12206 --k 5

# Export results to JSON
python search_sapbert_index.py \
    --query "diabetes" \
    --k 10 \
    --export diabetes_results.json
```

## Index Types Comparison

| Type | Accuracy | Speed | Memory | Best For |
|------|----------|-------|---------|----------|
| **Flat** | Highest | Slowest | High | Small datasets (<100K), maximum precision |
| **IVF** | High | Medium | Medium | Most datasets (100K-1M), balanced performance |
| **HNSW** | Good | Fastest | Low | Large datasets (>1M), speed priority |

## Python API Usage

### Index Creation

```python
from create_sapbert_index import SAPBERTIndexCreator

# Initialize creator
creator = SAPBERTIndexCreator()

# Create index
index_path = creator.create_index_from_csv(
    csv_path="data/entities.csv",
    output_dir="indexes",
    index_name="my_index",
    index_type="IVF",
    batch_size=16
)

print(f"Index created at: {index_path}")
```

### Searching

```python
from search_sapbert_index import SAPBERTIndexSearcher

# Initialize searcher
searcher = SAPBERTIndexSearcher()

# Load index
config = searcher.load_index("indexes/my_index")

# Single search
results = searcher.search("diabetes", k=10)
for result in results:
    print(f"{result['qid']}: {result['aliases'][0]} (Score: {result['similarity_score']:.3f})")

# Batch search
queries = ["diabetes", "cancer", "heart disease"]
batch_results = searcher.batch_search(queries, k=5)

# Get entity by QID
entity = searcher.get_entity_by_qid("Q12206")

# Find similar entities
similar = searcher.get_similar_entities("Q12206", k=5)

# Get statistics
stats = searcher.get_index_stats()
print(f"Index contains {stats['num_entities']} entities")
```

### Search Results Format

```python
# Each result is a dictionary with:
{
    'qid': 'Q12206',
    'aliases': ['diabetes', 'diabetes mellitus', 'DM'],
    'original_aliases': 'diabetes||diabetes mellitus||DM',
    'processed_text': 'diabetes || diabetes mellitus || DM',
    'similarity_score': 0.8542,
    'rank': 1,
    'index_id': 42
}
```

## Data Preprocessing

The utility automatically:

1. **Filters aliases**: Removes empty aliases and QID matches
2. **Cleans text**: Trims whitespace and handles special characters
3. **Validates entries**: Skips rows without valid aliases
4. **Combines aliases**: Joins aliases with " || " for embedding generation

## Performance Guidelines

### For Index Creation:
- Use GPU if available (automatically detected)
- Increase `batch_size` for faster processing (if you have enough memory)
- Use `IVF` index type for best balance of accuracy and speed
- Process large files in chunks if memory is limited

### For Searching:
- Load the index once and reuse for multiple searches
- Use `batch_search()` for multiple queries
- Adjust `k` parameter based on your needs
- Use appropriate index type for your dataset size

## Troubleshooting

### Common Issues:

**"CUDA out of memory"**
```bash
# Reduce batch size
python create_sapbert_index.py --batch_size 8
```

**"No valid aliases found"**
- Check your CSV format (ensure `qid` and `aliases` columns exist)
- Verify aliases are separated by `||`
- Ensure aliases are not empty or just QIDs

**"Index file not found"**
- Make sure you're using the correct path (without file extension)
- Check that all index files exist in the directory
- Use `--verbose` for detailed error messages

**"Model loading fails"**
```bash
# Use different model or check internet connection
python create_sapbert_index.py --model_name "dmis-lab/biobert-base-cased-v1.1"
```

### Memory Requirements:

- **Small dataset** (1K-10K entities): 2-4 GB RAM
- **Medium dataset** (10K-100K entities): 4-8 GB RAM  
- **Large dataset** (100K+ entities): 8+ GB RAM

## Workflow Examples

### Complete Workflow

```bash
# 1. Validate your CSV
python create_sapbert_index.py --csv_path my_data.csv --validate_only

# 2. Create index
python create_sapbert_index.py \
    --csv_path my_data.csv \
    --index_name medical_entities \
    --index_type IVF

# 3. Test search
python search_sapbert_index.py \
    --index_path ./indexes/medical_entities \
    --query "heart disease" \
    --k 5

# 4. Get statistics
python search_sapbert_index.py \
    --mode stats \
    --index_path ./indexes/medical_entities

# 5. Batch search and export
python search_sapbert_index.py \
    --mode batch_search \
    --index_path ./indexes/medical_entities \
    --queries "diabetes,cancer,asthma" \
    --export batch_results.json
```

### Production Deployment

```bash
# Create optimized index for production
python create_sapbert_index.py \
    --csv_path production_entities.csv \
    --output_dir /opt/indexes \
    --index_name production_index \
    --index_type HNSW \
    --batch_size 64

# Search in production
python search_sapbert_index.py \
    --index_path /opt/indexes/production_index \
    --query "user_search_term" \
    --k 20 \
    --export /tmp/search_results.json
```

## Help and Documentation

Get detailed help for any command:

```bash
python create_sapbert_index.py --help
python search_sapbert_index.py --help
```

## Citation

If you use this utility in your research, please cite:

```bibtex
@article{liu2021self,
  title={Self-Alignment Pretraining for Biomedical Entity Representations},
  author={Liu, Fangyu and Vulić, Ivan and Korhonen, Anna and Collier, Nigel},
  journal={arXiv preprint arXiv:2010.11784},
  year={2021}
}
```

## License

This utility is provided as-is for research and educational purposes. Please check the licenses of the underlying models (SAPBERT, Transformers, FAISS) for commercial usage.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify your CSV format and file paths
3. Ensure all dependencies are properly installed
4. Use `--verbose` flag for detailed error messages
5. Check available memory and GPU resources