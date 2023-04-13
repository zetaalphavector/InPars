import os
import re
import csv
import json
import random
import argparse
import subprocess
import statistics
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from transformers import set_seed
from pyserini.search.lucene import LuceneSearcher
from .dataset import load_corpus


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--dataset_source', default='ir_datasets',
                        help="The dataset source: ir_datasets or pyserini")
    parser.add_argument('--index', type=str, default='msmarco-passage')
    parser.add_argument('--max_hits', type=int, default=1000)
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument("--batch_size", default=32, type=int,
                        help="Batch size retrieval.")
    parser.add_argument('--threads', type=int, default=12)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    set_seed(args.seed)

    index = args.index
    if os.path.exists(args.dataset):
        if args.dataset.endswith('.csv'):
            corpus = pd.read_csv(args.input)
        else:
            corpus = pd.read_json(args.input, lines=True)
    else:
        corpus = load_corpus(args.dataset, source=args.dataset_source)
        index = f'beir-v1.0.0-{args.dataset}-flat'

    # Convert to {'doc_id': 'text'} format
    corpus = dict(zip(corpus['doc_id'], corpus['text']))

    if os.path.isdir(index):
        searcher = LuceneSearcher(index)
    else:
        searcher = LuceneSearcher.from_prebuilt_index(index)

    n_no_query = 0
    n_docs_not_found = 0
    n_examples = 0
    queries = []
    with open(args.input) as f, open(f'{Path(args.output).parent}/topics-{args.dataset}.tsv', 'w') as out:
        tsv_writer = csv.writer(out, delimiter='\t', lineterminator='\n')
        for (i, line) in enumerate(f):
            row = json.loads(line.strip())

            if not row['query']:
                n_no_query += 1
                continue

            query = ' '.join(row["query"].split())  # Removes line breaks and tabs.
            queries.append((query, None, row['doc_id']))
            tsv_writer.writerow([i, query])

    tmp_run = f'{Path(args.output).parent}/tmp-run-{args.dataset}.txt'
    if not os.path.exists(tmp_run):
        subprocess.run([
            'python3', '-m', 'pyserini.search.lucene',
                '--threads', '8',
                '--batch-size', str(args.batch_size),
                '--index', index,
                '--topics', f'{Path(args.output).parent}/topics-{args.dataset}.tsv',
                '--output', tmp_run,
                '--bm25',
        ])

    results = {}
    with open(tmp_run) as f:
        for line in f:
            qid, _, docid, rank, score, ranker = re.split(r"\s+", line.strip())
            if qid not in results:
                results[qid] = []
            results[qid].append(docid)

    with open(args.output, 'w') as fout:
        writer = csv.writer(fout, delimiter='\t', lineterminator='\n', quoting=csv.QUOTE_MINIMAL)
        for qid in tqdm(results, desc='Sampling'):
            hits = results[qid]
            query, log_probs, pos_doc_id = queries[int(qid)]
            n_examples += 1
            sampled_ranks = random.sample(range(len(hits)), min(len(hits), args.n_samples + 1))
            n_samples_so_far = 0
            for (rank, neg_doc_id) in enumerate(hits):
                if rank not in sampled_ranks:
                    continue

                if neg_doc_id not in corpus:
                    n_docs_not_found += 1
                    continue

                pos_doc_text = corpus[pos_doc_id].replace('\n', ' ').strip()
                neg_doc_text = corpus[neg_doc_id].replace('\n', ' ').strip()

                writer.writerow([query, pos_doc_text, neg_doc_text])
                n_samples_so_far += 1
                if n_samples_so_far >= args.n_samples:
                    break

    if n_no_query > 0:
        print(f'{n_no_query} lines without queries.')

    if n_docs_not_found > 0:
        print(f'{n_docs_not_found} docs returned by the search engine but not found in the corpus.')

    print("Done!")
