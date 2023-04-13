import os
import argparse
import pandas as pd
from .dataset import load_corpus
from transformers import set_seed
from .inpars import InPars

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', default='EleutherAI/gpt-j-6B')
    parser.add_argument('--prompt', type=str, default="inpars",
                        help="Prompt type to be used during query generation: \
                        inpars, promptagator or custom")
    parser.add_argument('--dataset', default=None, required=True,
                        help="Dataset from BEIR or custom corpus file (CSV or JSONL)")
    parser.add_argument('--dataset_source', default='ir_datasets',
                        help="The dataset source: ir_datasets or pyserini")
    parser.add_argument('--n_fewshot_examples', type=int, default=3)
    parser.add_argument('--max_doc_length', default=256, type=int, required=False)
    parser.add_argument('--max_query_length', default=200, type=int, required=False)
    parser.add_argument('--max_prompt_length', default=2048, type=int, required=False)
    parser.add_argument('--max_new_tokens', type=int, default=64)
    parser.add_argument('--max_generations', type=int, default=100_000)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--revision', type=str, default=None)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--int8', action='store_true')
    parser.add_argument('--torch_compile', action='store_true')
    parser.add_argument('--tf', action='store_true')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--device', type=str, default=None)
    # parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    set_seed(args.seed)

    if os.path.exists(args.dataset):
        if args.dataset.endswith('.csv'):
            dataset = pd.read_csv(args.input)
        else:
            dataset = pd.read_json(args.input, lines=True)
    else:
        dataset = load_corpus(args.dataset, args.dataset_source)

    if args.max_generations > len(dataset):
        args.max_generations = len(dataset)

    dataset = dataset.sample(args.max_generations)

    if args.n_fewshot_examples >= len(dataset):
        raise Exception(
            f'Number of few-shot examples must be higher than the number of documents \
            ({args.n_fewshot_examples} >= {len(dataset)})'
        )

    generator = InPars(
        base_model=args.base_model,
        revision=args.revision,
        corpus=args.dataset,
        prompt=args.prompt,
        n_fewshot_examples=args.n_fewshot_examples,
        max_doc_length=args.max_doc_length,
        max_query_length=args.max_query_length,
        max_prompt_length=args.max_prompt_length,
        max_new_tokens=args.max_new_tokens,
        fp16=args.fp16,
        int8=args.int8,
        tf=args.tf,
        device=args.device,
        torch_compile=args.torch_compile,
        # verbose=args.verbose,
    )

    generated = generator.generate(
        documents=dataset['text'],
        doc_ids=dataset['doc_id'],
        batch_size=args.batch_size,
    )
    dataset['query'] = [example['query'] for example in generated]
    dataset['log_probs'] = [example['log_probs'] for example in generated]
    dataset['prompt_text'] = [example['prompt_text'] for example in generated]
    dataset['doc_id'] = [example['doc_id'] for example in generated]
    dataset['fewshot_examples'] = [example['fewshot_examples'] for example in generated]
    dataset.to_json(args.output, orient='records', lines=True)
