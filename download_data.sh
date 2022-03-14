# exit when any command fails
set -e

# MS MARCO
# Original links
#wget https://www.dropbox.com/s/5pqpcnlzlib2b3a/run.bm25.dev.small.tsv.gz
#wget https://www.dropbox.com/s/iyw98nof7omynst/queries.dev.small.tsv
#wget https://www.dropbox.com/s/ie27l0mzcjb5fbc/qrels.dev.small.tsv
#wget https://www.dropbox.com/s/m1n2wf80l1lb9j1/collection.tar.gz

export DATA=/datadrive/msmarco
mkdir -p $DATA

gsutil cp gs://project-1462/maritaca/collection.tsv $DATA
gsutil cp gs://project-1462/maritaca/collection_shuffled.tsv $DATA
gsutil cp gs://project-1462/maritaca/qrels.dev.small.tsv $DATA
gsutil cp gs://project-1462/maritaca/queries.dev.small.tsv $DATA
gsutil cp gs://project-1462/maritaca/run.bm25.dev.small.tsv $DATA
gsutil cp gs://project-1462/maritaca/run.bm25.dev.small.txt $DATA


# TREC-DL 2020
export DATA=/datadrive/dl20
mkdir -p $DATA

gsutil cp gs://project-1462/maritaca/dl20/topics_in_qrels.dl20.txt $DATA
wget https://raw.githubusercontent.com/castorini/anserini/master/src/main/resources/topics-and-qrels/qrels.dl20-passage.txt --directory-prefix $DATA
gsutil cp gs://project-1462/maritaca/dl20/run.bm25.dl20.txt $DATA


# Robust04
export DATA=/datadrive/robust04
mkdir -p $DATA

gsutil cp gs://project-1462/maritaca/robust04/topics.robust04.tsv $DATA
gsutil cp gs://project-1462/maritaca/exp71/collection.tsv $DATA
gsutil cp gs://project-1462/maritaca/exp71/collection_shuffled.tsv $DATA
wget https://raw.githubusercontent.com/castorini/anserini/master/src/main/resources/topics-and-qrels/topics.robust04.txt --directory-prefix $DATA
wget https://raw.githubusercontent.com/castorini/anserini/master/src/main/resources/topics-and-qrels/qrels.robust04.txt --directory-prefix $DATA
wget https://storage.googleapis.com/castorini/robust04/run.robust04.bm25.txt --directory-prefix $DATA
wget https://storage.googleapis.com/castorini/robust04/trec_disks_4_and_5_concat.txt --directory-prefix $DATA


# Natural Questions
export DATA=/datadrive/nq
mkdir -p $DATA

gsutil cp gs://project-1462/maritaca/nq/qrels_nq_beir_trec.tsv $DATA
gsutil cp gs://project-1462/maritaca/nq/queries.tsv $DATA
gsutil cp gs://project-1462/maritaca/nq/run.bm25_beir_nq-trec.txt $DATA
gsutil cp gs://project-1462/maritaca/nq/corpus_with_title.tsv $DATA
gsutil cp gs://project-1462/maritaca/nq/corpus_shuffled.tsv $DATA
gsutil cp gs://project-1462/maritaca/nq/corpus_no_title.tsv $DATA
gsutil cp gs://project-1462/maritaca/nq/corpus_no_title_shuffled.tsv $DATA
gsutil cp -r gs://project-1462/maritaca/nq/jsonl_collection $DATA


# TREC-COVID
export DATA=/datadrive/trec_covid
mkdir -p $DATA

gsutil cp gs://project-1462/maritaca/trec-covid/corpus_with_title.tsv $DATA
gsutil cp gs://project-1462/maritaca/trec-covid/corpus_shuffled.tsv $DATA
gsutil cp gs://project-1462/maritaca/trec-covid/corpus_no_title.tsv $DATA
gsutil cp gs://project-1462/maritaca/trec-covid/corpus_no_title_shuffled.tsv $DATA
gsutil cp gs://project-1462/maritaca/trec-covid/qrels.tsv $DATA
gsutil cp gs://project-1462/maritaca/trec-covid/queries.tsv $DATA
gsutil cp gs://project-1462/maritaca/trec-covid/run.bm25.trec-covid.txt $DATA
gsutil cp -r gs://project-1462/maritaca/trec-covid/jsonl_collection $DATA


# FiQA
export DATA=/datadrive/fiqa
mkdir -p $DATA
gsutil cp gs://project-1462/maritaca/fiqa/corpus.tsv $DATA
