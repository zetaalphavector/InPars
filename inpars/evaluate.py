import os
import json
import argparse
import subprocess
from pyserini.search import get_qrels_file
from .utils import TRECRun


def run_trec_eval(run_file, qrels_file, relevance_threshold=1):
    result = subprocess.run(
        [
            "python3",
            "-m",
            "pyserini.eval.trec_eval",
            "-c",
            f"-l {relevance_threshold}",
            "-mall_trec",
            qrels_file,
            run_file,
        ],
        stdout=subprocess.PIPE,
    )
    metrics = {}
    for line in result.stdout.decode("utf-8").split("\n"):
        for metric in [
            "recip_rank",
            "recall_1000",
            "num_q",
            "num_ret",
            "ndcg_cut_10",
            "ndcg_cut_20",
            "map",
            "P_20",
            "P_30",
        ]:
            # the space is to avoid getting metrics such as ndcg_cut_100 instead of ndcg_cut_10 as but start with ndcg_cut_10
            if line.startswith(metric + " ") or line.startswith(metric + "\t"):
                metrics[metric] = float(line.split("\t")[-1])
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=str)
    parser.add_argument("--dataset", default="msmarco")
    parser.add_argument("--qrels", default=None)
    parser.add_argument("--relevance_threshold", default=1)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.dataset == "msmarco":
        dataset_name = "msmarco-passage-dev-subset"
    else:
        dataset_name = f"beir-v1.0.0-{args.dataset}-test"

    qrels_file = get_qrels_file(dataset_name)

    if args.qrels and os.path.exists(args.qrels):
        qrels_file = args.qrels

    run_file = args.run
    if args.run.lower() == "bm25":
        run = TRECRun(args.dataset)
        run_file = run.run_file

    results = run_trec_eval(run_file, qrels_file, args.relevance_threshold)
    if args.json:
        print(json.dumps(results))
    else:
        for (metric, value) in sorted(results.items()):
            print(f"{metric}: {value}")
