import glob
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
        f"python ../create_msmarco_monot5_input.py --queries {dataset}/topics.beir-v1.0.0-{pyserini_name}.test.tsv --run general/runs/run.beir-v1.0.0-{pyserini_name}-flat.trec --corpus {dataset}/corpus.tsv  --t5_input {dataset}/pairs/query_doc_pairs.dev.small.txt --t5_input_ids {dataset}/pairs/query_doc_pair_ids.dev.small.tsv".split()
    )
    subprocess.run(
        f"split --suffix-length 3 --numeric-suffixes --lines 700000 {dataset}/pairs/query_doc_pairs.dev.small.txt {dataset}/pairs/query_doc_pairs.dev.small.txt".split()
    )
    subprocess.run(
        f"gsutil -m cp -r {dataset}/pairs gs://{args.gcp_path}/{dataset}/".split()
    )
    print(f"{dataset}: Finished inpars uploading")
    print(f"{dataset}: Starting inpars inference")

    num_splits = len(glob.glob(f"{dataset}/pairs/query_doc_pairs.dev.small.txt0*")) - 1
    print(
        f"bash pairs_evaluate_inpars.sh {args.gcp_path} {dataset} {num_splits} {args.checkpoint} {args.tpu_proj} {args.tpu_name} {args.variation}"
    )
    subprocess.run(
        f"bash pairs_evaluate_inpars.sh {args.gcp_path} {dataset} {num_splits} {args.checkpoint} {args.tpu_proj} {args.tpu_name} {args.variation}".split()
    )
