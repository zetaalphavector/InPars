import os
import csv
import torch
import argparse
import pandas as pd
from math import ceil, exp
from typing import List
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration
)
from . import utils
from .dataset import load_corpus, load_queries

# Based on https://github.com/castorini/pygaggle/blob/f54ae53d6183c1b66444fa5a0542301e0d1090f5/pygaggle/rerank/base.py#L63
prediction_tokens = {
    'castorini/monot5-small-msmarco-10k':   ['▁false', '▁true'],
    'castorini/monot5-small-msmarco-100k':  ['▁false', '▁true'],
    'castorini/monot5-base-msmarco':        ['▁false', '▁true'],
    'castorini/monot5-base-msmarco-10k':    ['▁false', '▁true'],
    'castorini/monot5-large-msmarco':       ['▁false', '▁true'],
    'castorini/monot5-large-msmarco-10k':   ['▁false', '▁true'],
    'castorini/monot5-base-med-msmarco':    ['▁false', '▁true'],
    'castorini/monot5-3b-med-msmarco':      ['▁false', '▁true'],
    'castorini/monot5-3b-msmarco-10k':      ['▁false', '▁true'],
    'castorini/monot5-3b-msmarco':          ['▁false', '▁true'],
    'unicamp-dl/mt5-base-en-msmarco':       ['▁no'   , '▁yes'],
    'unicamp-dl/mt5-base-mmarco-v2':        ['▁no'   , '▁yes'],
    'unicamp-dl/mt5-base-mmarco-v1':        ['▁no'   , '▁yes'],
}


class Reranker:
    def __init__(self, silent=False, batch_size=8, fp16=False, torchscript=False, device=None):
        self.silent = silent
        self.batch_size = batch_size
        self.fp16 = fp16
        self.torchscript = torchscript
        self.device = device

    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        return MonoT5Reranker(model_name_or_path, **kwargs)


class MonoT5Reranker(Reranker):
    name: str = 'MonoT5'

    def __init__(self, model_name_or_path='castorini/monot5-base-msmarco-10k', token_false=None, token_true=True, **kwargs):
        super().__init__(**kwargs)
        if not self.device:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_args = {}
        if self.fp16:
            model_args["torch_dtype"] = torch.float16
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, **model_args)
        self.model = torch.compile(self.model)
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.token_false_id, self.token_true_id = self.get_prediction_tokens(
            model_name_or_path, self.tokenizer, token_false, token_true,
        )

    def get_prediction_tokens(self, model_name_or_path, tokenizer, token_false=None, token_true=None):
        if not (token_false and token_true):
            if model_name_or_path in prediction_tokens:
                token_false, token_true = prediction_tokens[model_name_or_path]
                token_false_id = tokenizer.get_vocab()[token_false]
                token_true_id  = tokenizer.get_vocab()[token_true]
                return token_false_id, token_true_id
            else:
                # raise Exception(f"We don't know the indexes for the non-relevant/relevant tokens for\
                #         the checkpoint {model_name_or_path} and you did not provide any.")
                return self.get_prediction_tokens('castorini/monot5-base-msmarco', self.tokenizer)
        else:
            token_false_id = tokenizer.get_vocab()[token_false]
            token_true_id  = tokenizer.get_vocab()[token_true]
            return token_false_id, token_true_id

    @torch.no_grad()
    def rescore(self, pairs: List[List[str]]):
        scores = []
        for batch in tqdm(
            utils.chunks(pairs, self.batch_size),
            disable=self.silent,
            desc="Rescoring",
            total=ceil(len(pairs) / self.batch_size),
        ):
            prompts = [f"Query: {query} Document: {text} Relevant:" for (query, text) in batch]
            tokens = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.tokenizer.model_max_length,
            ).to(self.device)
            output = self.model.generate(
                **tokens,
                max_new_tokens=1,
                return_dict_in_generate=True,
                output_scores=True,
            )
            batch_scores = output.scores[0]
            batch_scores = batch_scores[:, [self.token_false_id, self.token_true_id]]
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
            scores += batch_scores[:, 1].exp().tolist()
        return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='castorini/monot5-small-msmarco-100k',
            type=str, required=False, help="Reranker model.")
    parser.add_argument("--input_run", default=None, type=str,
                        help="Initial run to be reranked.")
    parser.add_argument("--output_run", default=None, type=str, required=True,
                        help="Path to save the reranked run.")
    parser.add_argument("--dataset", default=None, type=str,
                        help="Dataset name from BEIR collection.")
    parser.add_argument('--dataset_source', default='ir_datasets',
                        help="The dataset source: ir_datasets or pyserini")
    parser.add_argument("--corpus", default=None, type=str,
                        help="Document collection `doc_id` and `text` fields in CSV format.")
    parser.add_argument("--queries", default=None, type=str,
                        help="Queries collection with `query_id` and `text` fields in CSV format.")
    parser.add_argument("--device", default=None, type=str,
                        help="CPU or CUDA device.")
    parser.add_argument("--fp16", action="store_true",
                        help="Whether to use FP16 weights during inference.")
    parser.add_argument("--batch_size", default=16, type=int,
                        help="Batch size for inference.")
    parser.add_argument("--top_k", default=1_000, type=int,
                        help="Top-k documents to be reranked for each query.")
    args = parser.parse_args()

    if args.dataset:
        corpus = load_corpus(args.dataset, source=args.dataset_source)
        queries = load_queries(args.dataset, source=args.dataset_source)
    else:
        corpus = pd.read_csv(args.corpus)
        queries = pd.read_csv(args.queries)
        queries = dict(zip(queries['query_id'], queries['text']))
    
    # Convert to {'doc_id': 'text'} format
    corpus = dict(zip(corpus['doc_id'], corpus['text']))

    input_run = args.input_run
    if args.dataset and not args.input_run:
        input_run = args.dataset

    model = Reranker.from_pretrained(
        model_name_or_path=args.model,
        batch_size=args.batch_size,
        fp16=args.fp16,
        device=args.device,
        # torchscript=args.torchscript,
    )

    run = utils.TRECRun(input_run)
    run.rerank(model, queries, corpus, top_k=args.top_k)
    run.save(args.output_run)
