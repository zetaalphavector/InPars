#for inPars finetuned
GCP_PATH=$1
DATASET=$2
TPU_PROJ=$3
TPU_NAME=$4
VARIATION=$5
t5_mesh_transformer  \
  --tpu="${TPU_NAME}" \
  --gcp_project="${TPU_PROJ}" \
  --tpu_zone="europe-west4-a" \
  --model_dir="gs://${GCP_PATH}/${DATASET}/${VARIATION}/t5/3B/" \
  --gin_param="init_checkpoint = 'gs://castorini/monot5/experiments/3B/model.ckpt-1010000'" \
  --gin_file="dataset.gin" \
  --gin_file="models/bi_v1.gin" \
  --gin_file="gs://t5-data/pretrained_models/3B/operative_config.gin" \
  --gin_param="utils.tpu_mesh_shape.model_parallelism = 8" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '2x2'" \
  --gin_param="utils.run.train_dataset_fn = @t5.models.mesh_transformer.tsv_dataset_fn" \
  --gin_param="tsv_dataset_fn.filename = 'gs://${GCP_PATH}/${DATASET}/${VARIATION}/triples/synth.query_doc_pairs.train.tsv'" \
  --gin_file="learning_rate_schedules/constant_0_001.gin" \
  --gin_param="run.train_steps = 1010156" \
  --gin_param="run.save_checkpoints_steps = 156" \
  --gin_param="utils.run.batch_size=('tokens_per_batch', 65536)" \
  --gin_param="serialize_num_microbatches.tokens_per_microbatch_per_replica = 1024" \
  --gin_param="utils.run.sequence_length = {'inputs': 512, 'targets': 2}" \
  --gin_param="Bitransformer.decode.max_decode_length = 2"