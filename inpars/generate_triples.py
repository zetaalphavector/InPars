import os
import csv
import json
import random
import argparse
import statistics
import pandas as pd
from tqdm import tqdm
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
    parser.add_argument("--device", default=None, type=str,
                        help="CPU or CUDA device.")
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use FP16 weights during inference.")
    parser.add_argument("--batch_size", default=16, type=int,
                        help="Batch size for inference.")
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--threads', type=int, default=12)
    parser.add_argument('--source', default='ir_datasets',
                        help="The dataset source: ir_datasets or pyserini")
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
        corpus = load_corpus(args.dataset, args.source, source=args.dataset_source)
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
    with open(args.input) as f:
        for line in f:
            row = json.loads(line.strip())

            if not row['query']:
                n_no_query += 1
                continue

            query = ' '.join(row["query"].split())  # Removes line breaks and tabs.
            queries.append((query, None, row['doc_id']))

    print('Retrieving candidates...')
    results = searcher.batch_search(
        [q[0] for q in queries],
        list(map(str, range(len(queries)))),
        threads=args.threads,
        k=args.max_hits + 1,
    )

    with open(args.output, 'w') as fout:
        writer = csv.writer(fout, delimiter='\t', lineterminator='\n')
        for qid in tqdm(results, desc='Sampling'):
            hits = results[qid]
            query, log_probs, pos_doc_id = queries[int(qid)]
            n_examples += 1
            sampled_ranks = random.sample(range(len(hits)), min(len(hits), args.n_samples + 1))
            n_samples_so_far = 0
            for rank, hit in enumerate(hits):
                neg_doc_id = hit.docid

                if rank not in sampled_ranks:
                    continue

                if neg_doc_id not in corpus:
                    n_docs_not_found += 1
                    continue

                pos_doc_text = corpus[pos_doc_id].strip()
                neg_doc_text = corpus[neg_doc_id].strip()

                writer.writerow([query, pos_doc_text, neg_doc_text])
                n_samples_so_far += 1
                if n_samples_so_far >= args.n_samples:
                    break

    if n_no_query > 0:
        print(f'{n_no_query} lines without queries.')

    if n_docs_not_found > 0:
        print(f'{n_docs_not_found} docs returned by the search engine but not found in the corpus.')

    print("Done!")
