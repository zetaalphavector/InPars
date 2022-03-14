import argparse
import json
import numpy as np
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True,
                        help='Path for the output file.')
    parser.add_argument('--top_k', type=int, default=50_000,
                        help='Write only top_k best scored query-doc pairs.')
    parser.add_argument('--min_tokens', type=int, default=3,
                        help='Skip question that have fewer than this number of words.')
    parser.add_argument('--max_tokens', type=int, default=10000000,
                        help='Skip question that have more than this number of words.')
    parser.add_argument('--scoring_function', type=str, default='mean_log_probs',
                        help='Possible values are "sum_log_probs", "mean_log_probs" and "mean_probs"')
    parser.add_argument('--skip_questions_copied_from_context', action='store_true',
                        help='If passed, skip questions that were copied from the passage.')

    args = parser.parse_args()

    assert args.scoring_function in ['mean_probs', 'mean_log_probs', 'sum_log_probs', 'len']

    with open(args.input) as f:
        rows = []
        for line in tqdm(f):
    
            row = json.loads(line.strip())
            if len(row['log_probs']) < args.min_tokens:
                continue
            if len(row['log_probs']) > args.max_tokens:
                continue
            if args.skip_questions_copied_from_context:
                if row['question'].lower() in row['doc_text'].lower():
                    continue

            if args.scoring_function == 'mean_probs':
                row['log_probs'] = np.exp(row['log_probs'])

            if args.scoring_function == 'sum_log_probs':
                row['log_probs'] = np.sum(row['log_probs'])

            if args.scoring_function == 'len':
                if len(row['question']) < 10 or len(row['question']) > 200 or 'ITAG' in row['doc_text'] or len(row['doc_text']) < 300:
                    continue
            else:
                row['log_probs'] = np.mean(row['log_probs'])

            rows.append(row)
    rows.sort(key=lambda row: row['log_probs'], reverse=True)

    with open(args.output, 'w') as fout:
        for row in rows[:args.top_k]:
            fout.write(json.dumps(row) + '\n')

    print('Done!')