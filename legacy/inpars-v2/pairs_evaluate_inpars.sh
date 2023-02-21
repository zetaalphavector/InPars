#for inPars finetuned
GCP_PATH=$1
DATASET=$2
NUM=$3
CHECKPOINT=$4
TPU_PROJ=$5
TPU_NAME=$6
VARIATION=$7
for ITER in `seq -f "%03g" 0 $NUM`; do
    echo "Running $DATASET iter: $ITER" >> out.log_eval_exp
    t5_mesh_transformer \
    --tpu="${TPU_NAME}" \
    --gcp_project="${TPU_PROJ}" \
    --tpu_zone="europe-west4-a" \
    --model_dir="gs://${GCP_PATH}/${DATASET}/${VARIATION}/t5/3B/" \
    --gin_file="gs://t5-data/pretrained_models/3B/operative_config.gin" \
    --gin_file="infer.gin" \
    --gin_file="beam_search.gin" \
    --gin_param="utils.tpu_mesh_shape.tpu_topology = '2x2'" \
    --gin_param="infer_checkpoint_step = ${CHECKPOINT}" \
    --gin_param="utils.run.sequence_length = {'inputs': 512, 'targets': 2}" \
    --gin_param="Bitransformer.decode.max_decode_length = 2" \
    --gin_param="input_filename = 'gs://${GCP_PATH}/${DATASET}/pairs/query_doc_pairs.dev.small.txt${ITER}'" \
    --gin_param="output_filename = 'gs://${GCP_PATH}/${DATASET}/${VARIATION}/score/query_doc_pair_scores.dev.small.txt${ITER}'" \
    --gin_param="utils.run.batch_size=('tokens_per_batch', 65536)" \
    --gin_param="Bitransformer.decode.beam_size = 1" \
    --gin_param="Bitransformer.decode.temperature = 0.0" \
    --gin_param="Unitransformer.sample_autoregressive.sampling_keep_top_k = -1" \
    --gin_param="utils.tpu_mesh_shape.model_parallelism = 8"
done