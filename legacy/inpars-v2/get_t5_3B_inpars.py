import os
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
    "--checkpoint",
    type=str,
    default="1010156",
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


for dataset, pyserini_name in zip(datasets, pyserini_names):
    subprocess.run(
        f"mkdir -p {dataset}/{args.variation}/score".split(), stderr=subprocess.PIPE
    )
    subprocess.run(
        f"gsutil cp -r gs://{args.gcp_path}/{dataset}/{args.variation}/score {dataset}/{args.variation}/".split(),
        stderr=subprocess.PIPE,
    )
    os.system(
        f"cat {dataset}/{args.variation}/score/query_doc_pair_scores.dev.small.txt???-{args.checkpoint} > {dataset}/{args.variation}/score/query_doc_pair_scores.dev.small.txt-{args.checkpoint}"
    )
    subprocess.run(
        f"python ../convert_run_from_t5_to_trec_format.py     --predictions={dataset}/{args.variation}/score/query_doc_pair_scores.dev.small.txt-{args.checkpoint}     --query_run_ids={dataset}/pairs/query_doc_pair_ids.dev.small.tsv     --output={dataset}/runs/run.T5-{args.variation}-10k.txt".split(),
        stderr=subprocess.PIPE,
    )
    print(dataset + ":")
    print(
        subprocess.run(
            f"ir_measures {dataset}/qrels.beir-v1.0.0-{pyserini_name}.test.txt {dataset}/runs/run.T5-{args.variation}-10k.txt nDCG@10 nDCG@20 AP".split(),
            capture_output=True,
            text=True,
        ).stdout
    )
