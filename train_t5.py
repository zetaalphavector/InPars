import argparse
import collections
import os
import pandas as pd
import subprocess
import time
import wandb

from tqdm.auto import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from pygaggle.rerank.base import Query, Text
from pygaggle.rerank.transformer import MonoT5
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


class MonoT5Dataset(Dataset):
    def __init__(self, data, queries=None, corpus=None, qrels=None, training: bool=False):
        self.data = data
        self.queries = queries
        self.corpus = corpus
        self.qrels = qrels
        self.training = training

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        if self.training:
            query, document, label = sample
        else:
            query_id, doc_id, _ = sample
            query = self.queries[query_id]
            document = self.corpus[doc_id]
            label = 'false'
            if doc_id in self.qrels[query_id]:
                if self.qrels[query_id][doc_id] > 0:
                    label = 'true'
        text = f'Query: {query} Document: {document} Relevant:'
        return {
          'text': text,
          'labels': label,
        }


def run_trec_eval(run_file, qrels_file, relevance_threshold=1):
    result = subprocess.run(
        ['python', '-m', 'pyserini.eval.trec_eval', f'-l {relevance_threshold}', '-mall_trec', qrels_file, run_file],
        stdout=subprocess.PIPE,
    )
    metrics = {}
    print(result.stdout.decode('utf-8'))
    for line in result.stdout.decode('utf-8').split('\n'):
        for metric in ['recip_rank', 'recall_1000', 'num_q', 'num_ret', 'ndcg_cut_10', 'ndcg_cut_20', 'map', 'P_20', 'P_30']:
            # the space is to avoid getting metrics such as ndcg_cut_100 instead of ndcg_cut_10 as but start with ndcg_cut_10
            if line.startswith(metric + ' '): 
                metrics[metric] = float(line.split('\t')[-1])
    return metrics


def run_msmarco_eval(run_file, qrels_file):
    result = subprocess.run(
        ['python', '../msmarco_passage_eval.py', qrels_file, run_file],
        stdout=subprocess.PIPE,
    )
    print(result.stdout.decode('utf-8'))
    num_total_queries = len(set([line.strip().split()[0] for line in open(qrels_file)]))

    for line in result.stdout.decode('utf-8').split('\n'):
        if line.startswith('MRR @10: '):
            mrr10 = float(line.replace('MRR @10: ', ''))
        if line.startswith('QueriesRanked: '):
            n_queries = float(line.replace('QueriesRanked: ', ''))

    return {'mrr@10': mrr10 * num_total_queries / n_queries, 'n_queries_msmarco': n_queries}


def rerank(model, output_dir, run, queries, corpus, version=''):
    results = []
    for query_id, docid, rank in run:
        results.append({
            'query_id': query_id,
            'docid': docid,
            'rank': rank,
            'query': queries[query_id],
            'text': corpus[docid],
        })

    df_results = pd.DataFrame(results)

    reranked_run_trec_file = os.path.join(output_dir, f'reranked_run_{version}.txt')
    reranked_run_msmarco_file = os.path.join(output_dir, f'reranked_run_{version}.tsv')

    with open(reranked_run_trec_file, 'w') as fout_trec, open(reranked_run_msmarco_file, 'w') as fout_marco:
        for query_id, group in tqdm(df_results.groupby('query_id'), mininterval=0.5):
            query = Query(group.iloc[0]['query'])

            texts = [Text(i.text, {'docid': i.docid}, 0) for i in group.itertuples()]
            reranked = model.rerank(query, texts)

            for i in range(len(reranked)):
                docid = reranked[i].metadata['docid']
                score = reranked[i].score
                fout_trec.write(f'{query_id} Q0 {docid} {i + 1} {score} Anserini\n')
                fout_marco.write(f'{query_id}\t{docid}\t{i + 1}\n')

    return reranked_run_trec_file, reranked_run_msmarco_file


def compute_metrics(output_dir, qrels_file, run, queries, corpus, batch_size, relevance_threshold=1):
    def calculate(_):
        print('Evaluating...')
        model = MonoT5('castorini/monot5-base-msmarco', use_amp=True)
        model.tokenizer = MonoT5.get_tokenizer('t5-base', batch_size=batch_size)
        model.model = trainer.model.to(torch.device('cuda'))

        results = {
            'step': trainer.state.global_step,
            'epoch': int(trainer.state.epoch),
        }

        reranked_run_trec_file, reranked_run_msmarco_file = rerank(
            model=model,
            output_dir=output_dir,
            run=run,
            queries=queries,
            corpus=corpus,
            version=f'{trainer.state.global_step}',
        )

        results.update(run_trec_eval(reranked_run_trec_file, qrels_file, relevance_threshold=relevance_threshold))
        results.update(run_msmarco_eval(reranked_run_msmarco_file, qrels_file))

        if os.path.exists(f'{output_dir}/results.csv'):
            df_results = pd.read_csv(f'{output_dir}/results.csv')
            df_results = df_results.append(results, ignore_index=True)
        else:
            df_results = pd.DataFrame(results, index=[0])

        df_results.to_csv(f'{output_dir}/results.csv', index=False)

        return results

    return calculate


def get_optimizer(model, optimizer_name, scheduler_name, lr, weight_decay, step_size, gamma):

    optimizer = getattr(torch.optim, optimizer_name)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        },
    ]
    optimizer = optimizer(optimizer_grouped_parameters, lr=lr, weight_decay=weight_decay)

    print(f'=> Using {optimizer_name} optimizer')

    if scheduler_name == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma)
        print(f'=> Using StepLR (step_size = {step_size}, gamma = {gamma})')
    else:
        raise Exception(f'Scheduler not implemented: {scheduler_name}')

    return optimizer, scheduler


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', default='t5-base', type=str,
                        help='Base model to fine tune.')
    parser.add_argument('--run', type=str, default='./run.bm25.synthetic.t0pp.txt')
    parser.add_argument('--queries', type=str, default='./queries.synthetic.t0pp.tsv')
    parser.add_argument('--corpus', type=str, default='./collection.tsv')
    parser.add_argument('--qrels', type=str, default='./qrels.synthetic.t0pp.txt')
    parser.add_argument('--relevance_threshold', type=int, default=1)
    parser.add_argument('--triples_train', type=str, default='./triples.train.tsv')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path for trained model and checkpoints.')
    parser.add_argument('--save_every_n_steps', default=100, type=int,
                        help='Save every N steps. (recommended 10000)')
    parser.add_argument('--eval_steps', default=100, type=int,
                        help='Evaluation steps.')
    parser.add_argument('--logging_steps', default=10, type=int,
                        help='Logging steps.')
    parser.add_argument('--per_device_train_batch_size', default=8, type=int,
                        help='Per device batch size.')
    parser.add_argument('--gradient_accumulation_steps', default=16, type=int,
                        help='Gradient accumulation.')
    parser.add_argument('--per_device_eval_batch_size', default=16, type=int,
                        help='Eval batch size.')
    parser.add_argument('--learning_rate', default=3e-4, type=float,
                        help='Learning rate.')
    parser.add_argument('--epochs', default=1, type=int,
                        help='Number of epochs to train')
    parser.add_argument('--max_eval_queries', type=int, default=-1)
    parser.add_argument('--max_eval_docs_per_query', type=int, default=100)

    device = torch.device('cuda')
    args = parser.parse_args()
    
    args.output_dir = os.path.join(args.output_dir, time.strftime("%Y%m%d-%H%M%S"))

    queries = {}
    with open(args.queries, 'r', encoding='utf8') as f:
        for line in tqdm(f, mininterval=0.5, desc='Loading Queries'):
            query_id, query_text = line.strip().split('\t')
            queries[query_id] = query_text
            if args.max_eval_queries > -1 and len(queries) >= args.max_eval_queries:
                break

    qrels = collections.defaultdict(lambda: collections.defaultdict(str))
    with open(args.qrels, 'r', encoding='utf8') as f:
        for line in tqdm(f, mininterval=0.5, desc='Loading Qrels'):
            query_id, _, doc_id, relevance = line.strip().split()
            qrels[query_id][doc_id] = int(relevance)

    run = []
    with open(args.run, 'r', encoding='utf8') as f:
        for line in tqdm(f, mininterval=0.5, desc='Loading Run'):
            items = line.strip().split()
            if len(items) == 3:
                query_id, doc_id, rank = items
            else:
                query_id, _, doc_id, rank, _, _  = items
            if query_id not in queries:
                continue
            rank = int(rank)
            if rank > args.max_eval_docs_per_query:
                continue
            run.append((query_id, doc_id, rank))

    corpus = {}
    with open(args.corpus, 'r', encoding='utf8') as f:
        for line in tqdm(f, mininterval=0.5, desc='Loading Corpus'):
            doc_id, doc_text = line.strip().split('\t')
            corpus[doc_id] = doc_text

    train_examples = []
    with open(args.triples_train, 'r', encoding='utf8') as f:
        for line in tqdm(f, mininterval=0.5, desc='Loading Triples Train'):
            query, pos_doc, neg_doc = line.split('\t')
            neg_doc = neg_doc.strip()
            if not query:
                raise Exception(f'Query "{query}" is empty')
            if pos_doc:
                train_examples.append((query, pos_doc, 'true'))
            if neg_doc:
                train_examples.append((query, neg_doc, 'false'))

    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)
    tokenizer = AutoTokenizer.from_pretrained('t5-base')

    train_dataset = MonoT5Dataset(data=train_examples, training=True)
    # We create a dummy eval set with only two examples so calculate can be called.
    eval_dataset = MonoT5Dataset(data=run[:2], queries=queries, corpus=corpus, qrels=qrels, training=False)
    print('Total training data:', len(train_dataset))
    print('Total eval data:', len(eval_dataset))

    print('Training examples:')
    for i in range(2):
        print(train_dataset[i])

    print('Validation examples:')
    for i in range(2):
        print(eval_dataset[i])

    def smart_batching_collate_text_only(batch):
        texts = [example['text'] for example in batch]
        tokenized = tokenizer(texts, padding=True, truncation='longest_first', return_tensors='pt', max_length=512)
        tokenized['labels'] = tokenizer([example['labels'] for example in batch], return_tensors='pt')['input_ids']

        for name in tokenized:
            tokenized[name] = tokenized[name].to(device)

        return tokenized

    print('Training examples (loader):')
    for batch in DataLoader(train_dataset, batch_size=2, collate_fn=smart_batching_collate_text_only):
        print(batch)
        break

    wandb.login(key=os.getenv('WANDB_API_KEY'))

    train_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        do_train=True,
        do_eval=True,
        evaluation_strategy='steps',
        eval_steps=args.eval_steps,
        save_strategy='steps',
        save_steps=args.save_every_n_steps,
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=5e-5,
        num_train_epochs=args.epochs,
        warmup_steps=0,
        adafactor=False,
        disable_tqdm=False,
        load_best_model_at_end=False,
        predict_with_generate=True,
        dataloader_pin_memory=False,
        fp16=False,
        report_to='wandb',
        seed=2,
    )

    calculator = compute_metrics(
        output_dir=args.output_dir,
        qrels_file=args.qrels,
        queries=queries,
        run=run,
        corpus=corpus,
        batch_size=args.per_device_eval_batch_size,
        relevance_threshold=args.relevance_threshold,
    )

    optimizer, scheduler = get_optimizer(
        model=model,
        optimizer_name='AdamW',
        scheduler_name='StepLR',
        lr=args.learning_rate,
        weight_decay=5e-5,
        step_size=1000,
        gamma=1.0)

    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=smart_batching_collate_text_only,
        compute_metrics=calculator,
        optimizers=[optimizer, scheduler]
    )

    trainer.train()

    trainer.save_model(args.output_dir)
    trainer.save_state()