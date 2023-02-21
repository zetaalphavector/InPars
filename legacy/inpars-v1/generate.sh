# exit when any command fails
set -e

python generate_queries_openai.py --collection $1 --output $2 --engine $3


## Filter generated queries
python filter_queries_by_score.py --input $2 --output path/to/filtered/queries --scoring_function mean_log_probs


## Create training data
python generate_triples_train.py --input path/to/queries --output path/to/training/data --output_ids path/to/training/data/ids --corpus path/to/collection --index index/name/pyserini