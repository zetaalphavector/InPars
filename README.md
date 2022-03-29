# InPars: Inquisitive Parrots for Search [<img src="https://img.shields.io/badge/arXiv-2202.05144-b31b1b.svg">](https://https://arxiv.org/abs/2202.05144)

InPars is a simple yet effective approach towards efficiently using large LMs in retrieval tasks. For more information, checkout our paper:
    * [**InPars: Data Augmentation for Information Retrieval using Large Language Models**](https://arxiv.org/abs/2202.05144)

In this work, we use large LMs to generate labeled data in a few-shot manner for IR tasks.
We then finetune retrieval models on this synthetic data and use them to rerank the search results of a firs-stage retrieval system.

![Ilustration of our method](src/inpars.png)

Add a more complete description..

## How to Generate
To generate synthetic data: 
```
python generate_queries_openai.py --collection path/to/collection --output path/to/save/output --engine curie
```

## Filter generated queries
You can filter the generated queries by score. The score is the log probability assigned to each predicted token by the model.  
```
python filter_queries_by_score.py --input path/to/queries --output path/to/filtered/queries --scoring_function mean_log_probs
```

## Create training data
In order to create the training pairs, we use BM25 to sample negatives examples from collection.
```
python generate_triples_train.py --input path/to/queries --output path/to/training/data --output_ids path/to/training/data/ids --corpus path/to/collection --index index/name/pyserini
```

# Training

## monoT5-220M
```
python train_t5.py \
    --corpus=path/to/collection \
    --triples_train=$OUTPUT/triples.train.tsv \
    --queries=$DATA/path/to/eval/queries \
    --qrels=$DATA/path/to/eval/qrels/ \
    --run=$DATA/dl20/path/to/eval/run \
    --relevance_threshold=2 \
    --output_dir=$OUTPUT \
    --save_every_n_steps=156 \
    --eval_steps=156 \
    --max_eval_queries=54 \
    --max_eval_docs_per_query=1000 \
```
Add some text...

## monoT5-3B
### Generate training file

```
nohup t5_mesh_transformer  \
  --tpu="${TPU_NAME}" \
  --gcp_project="${PROJECT_NAME}" \
  --tpu_zone="europe-west4-a" \
  --model_dir="${GS_FOLDER}" \
  --gin_param="init_checkpoint = 'gs://t5-data/pretrained_models/3B/model.ckpt-1000000.meta'" \
  --gin_file="dataset.gin" \
  --gin_file="models/bi_v1.gin" \
  --gin_file="gs://t5-data/pretrained_models/3B/operative_config.gin" \
  --gin_param="utils.tpu_mesh_shape.model_parallelism = 8" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '2x2'" \
  --gin_param="utils.run.train_dataset_fn = @t5.models.mesh_transformer.tsv_dataset_fn" \
  --gin_param="tsv_dataset_fn.filename = 'gs://castorini/monot5/data/query_doc_pairs.train.tsv'" \
  --gin_file="learning_rate_schedules/constant_0_001.gin" \
  --gin_param="run.train_steps = 1100000" \
  --gin_param="run.save_checkpoints_steps = 10000" \
  --gin_param="utils.run.batch_size=('tokens_per_batch', 65536)" \
  >> out.log_exp 2>&1 &
  ```