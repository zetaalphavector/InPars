# exit when any command fails
set -e

# MS MARCO
# Original links
export DATA=data/msmarco
mkdir -p $DATA

wget -P $DATA https://msmarco.blob.core.windows.net/msmarcoranking/collection.tar.gz
wget -P $DATA https://raw.githubusercontent.com/castorini/anserini/master/src/main/resources/topics-and-qrels/topics.msmarco-passage.dev-subset.txt
wget -P $DATA https://raw.githubusercontent.com/castorini/anserini/master/src/main/resources/topics-and-qrels/qrels.msmarco-passage.dev-subset.txt
wget -P $DATA https://storage.googleapis.com/duobert_git/run.bm25.dev.small.tsv
tar -xf $DATA/collection.tar.gz -C $DATA
rm $DATA/collection.tar.gz
