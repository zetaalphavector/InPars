export PROJECT_NAME=nm-tpus-5
export PROJECT_ID=971648346294
export INSTANCE_NAME=rodrigo-1
export TPU_NAME=rodrigo-1

gcloud beta compute --project=${PROJECT_NAME} instances create ${INSTANCE_NAME} --zone=europe-west4-a --machine-type=n1-standard-4 --subnet=default --network-tier=PREMIUM --maintenance-policy=MIGRATE --service-account=${PROJECT_ID}-compute@developer.gserviceaccount.com  --scopes=https://www.googleapis.com/auth/cloud-platform --image=debian-10-buster-v20201112 --image-project=debian-cloud --boot-disk-size=25GB --boot-disk-type=pd-standard --boot-disk-device-name=${INSTANCE_NAME} --reservation-affinity=any

gcloud alpha compute tpus create ${TPU_NAME} --project ${PROJECT_NAME} --zone=europe-west4-a  --accelerator-type=v3-8  --version=2.8.0

python generate_triples_train.py \
    --input ./data/msmarco/synthetic_msmarco_davinci-002_gbq.jsonl \
    --output ./data/msmarco/synthetic.triples.train.tsv \
    --output_ids ./data/msmarco/synthetic.triples.train.ids.tsv \
    --corpus ./data/msmarco/collection_shuffled.tsv \
    --index msmarco-v1-passage