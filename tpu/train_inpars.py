import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--gcp_path",
    type=str,
    required=True,
)
parser.add_argument(
    "--variation",
    type=str,
    default="",
)
parser.add_argument(
    "--tpu_proj",
    type=str,
    required=True,
)
parser.add_argument(
    "--tpu_name",
    type=str,
    required=True,
)
args = parser.parse_args()

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
    subprocess.run(
        f"python ../filter_queries_by_score.py --input {dataset}/{synth} --output {dataset}/triples/filtered_triples.jsonl --scoring_function len".split()
    )
    subprocess.run(
        f"python ../generate_triples_train.py --input {dataset}/triples/filtered_triples.jsonl --output {dataset}/triples/synth.synthetic.triples.train.tsv --output_ids {dataset}/del_temp --corpus {dataset}/corpus.tsv --index beir-v1.0.0-{pyserini_name}-flat".split()
    )
    subprocess.run(
        f"python ../create_msmarco_monot5_train.py --triples_train {dataset}/triples/synth.synthetic.triples.train.tsv --output_to_t5 {dataset}/triples/synth.query_doc_pairs.train.tsv".split()
    )
    subprocess.run(
        f"gsutil cp {dataset}/triples/synth.query_doc_pairs.train.tsv gs://{args.gcp_path}/{dataset}/triples/".split()
    )
    subprocess.run(
        f"bash train_inpars.sh {args.gcp_path} {dataset} {args.tpu_proj} {args.tpu_name} {args.variation}".split()
    )
