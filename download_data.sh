# exit when any command fails
set -e

# MS MARCO
# Original links
export DATA=data/msmarco
mkdir -p $DATA

wget -P $DATA https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz
wget -P $DATA https://raw.githubusercontent.com/castorini/anserini/master/src/main/resources/topics-and-qrels/topics.msmarco-passage.dev-subset.txt
wget -P $DATA https://raw.githubusercontent.com/castorini/anserini/master/src/main/resources/topics-and-qrels/qrels.msmarco-passage.dev-subset.txt
gsutil cp gs://nm_datasets/trec_covid/runs/run.beir-v1.0.0-trec-covid-flat.trec $DATA
tar -xf $DATA/collection.tar.gz -C $DATA
rm $DATA/collection.tar.gz