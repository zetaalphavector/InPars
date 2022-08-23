from pathlib import Path
import wget
import gzip
import shutil
from pyserini.search.lucene import LuceneSearcher
import json

datasets = [
    "bioasq",
    "dbpedia",
    "fiqa",
    "nfcorpus",
    "robust04",
    "scidocs",
    "scifact",
    "signal",
    "touche",
    "trec_covid",
    "trecnews",
    "arguana",
    "climate_fever",
    "nq",
    "fever",
    "hotpotqa",
    "quora",
]
pyserini_names = [
    "bioasq",
    "dbpedia-entity",
    "fiqa",
    "nfcorpus",
    "robust04",
    "scidocs",
    "scifact",
    "signal1m",
    "webis-touche2020",
    "trec-covid",
    "trec-news",
    "arguana",
    "climate-fever",
    "nq",
    "fever",
    "hotpotqa",
    "quora",
]
synths = [
    "bioasq_synthetic_queries_100k.jsonl",
    "dbpedia_synthetic_queries_100k.jsonl",
    "fiqa_synthetic_queries_100k.jsonl",
    "nfcorpus_synthetic_queries_100k.jsonl",
    "robust04_synthetic_queries_100k.jsonl",
    "scidocs_synthetic_queries_100k.jsonl",
    "scifacts_synthetic_queries_100k.jsonl",
    "signal_synthetic_queries_100k.jsonl",
    "touche_synthetic_queries_100k.jsonl",
    "trec_covid_synthetic_queries_100k.jsonl",
    "trec_news_synthetic_queries_100k.jsonl",
    "arguana_synthetic_queries_100k.jsonl",
    "climate_fever_synthetic_queries_100k.jsonl",
    "nq_synthetic_queries_100k.jsonl",
    "fever_synthetic_queries_100k.jsonl",
    "hotpotqa_synthetic_queries_100k.jsonl",
    "quora_synthetic_queries_100k.jsonl",
]

for dataset, pyserini_name, synth in zip(datasets, pyserini_names, synths):
    Path(f"/{dataset}/pairs").mkdir(parents=True, exist_ok=True)
    Path(f"/{dataset}/triples").mkdir(parents=True, exist_ok=True)
    Path(f"/{dataset}/runs").mkdir(parents=True, exist_ok=True)
    wget.download(f"https://zav-public.s3.amazonaws.com/inpars/{synth}", f"{dataset}/")
    wget.download(
        f"https://huggingface.co/datasets/unicamp-dl/beir-runs/resolve/main/bm25/run.beir-v1.0.0-{pyserini_name}-flat.trec",
        f"{dataset}/runs/",
    )
    wget.download(
        f"https://github.com/castorini/anserini/blob/master/src/main/resources/topics-and-qrels/qrels.beir-v1.0.0-{pyserini_name}.test.txt",
        f"{dataset}/",
    )
    wget.download(
        f"https://github.com/castorini/anserini/raw/master/src/main/resources/topics-and-qrels/topics.beir-v1.0.0-{pyserini_name}.test.tsv.gz",
        f"{dataset}/",
    )

    with gzip.open(
        f"{dataset}/topics.beir-v1.0.0-{pyserini_name}.test.tsv.gz", "rb"
    ) as f_in:
        with open(
            f"{dataset}/topics.beir-v1.0.0-{pyserini_name}.test.tsv", "wb"
        ) as f_out:
            shutil.copyfileobj(f_in, f_out)

    searcher = LuceneSearcher.from_prebuilt_index(f"beir-v1.0.0-{pyserini_name}-flat")
    with open(f"{dataset}/corpus.tsv", "w") as writer:
        for i in range(searcher.num_docs):
            doc = searcher.doc(i).raw()
            doc = json.loads(doc)
            s = f'{str(doc["_id"])}\t{" ".join(doc["text"]).split()}\n'
            writer.write(s)
