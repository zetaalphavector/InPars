import torch
from functools import partial
from datasets import load_dataset
from dataclasses import dataclass, field
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    Trainer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorWithPadding,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    set_seed,
)

monot5_prompt = "Query: {query} Document: {text} Relevant:"
flan_prompt = """Is the following passage relevant to the query?
Query: {query}
Passage: {text}"""

monot5_outputs = ['false', 'true']
flant5_outputs = ['no', 'yes']


@dataclass
class ExtraArguments:
    triples: str = field(
        default=None,
        metadata={
            "help": "Triples file containing query, positive and negative examples (TSV format)."
        },
    )
    pairs: str = field(
        default=None,
        metadata={
            "help": "File containing pairs of query, passage and label (positive/negative) in TSV format."
        },
    )
    base_model: str = field(
        default="t5-base",
        metadata={
            "help": "Base model to fine-tune."
        },
    )
    max_doc_length: int = field(
        default=300,
        metadata={
            "help": "Maximum document length. Documents exceding this length will be truncated."
        },
    )


def split_triples(triples, seq2seq=True):
    if seq2seq:
        examples = {
            'label': [],
            'text': [],
        }
        for i in range(len(triples['query'])):
            examples['text'].append(prompt_template.format(query=triples["query"][i], text=triples["positive"][i]))
            examples['label'].append(token_true)
            examples['text'].append(prompt_template.format(query=triples["query"][i], text=triples["negative"][i]))
            examples['label'].append(token_false)
    else:
        examples = {
            'label': [],
            'query': [],
            'text': [],
        }
        for i in range(len(triples['query'])):
            examples['query'].append(triples['query'][i])
            examples['text'].append(triples['positive'][i])
            examples['label'].append(token_true)
            examples['query'].append(triples['query'][i])
            examples['text'].append(triples['negative'][i])
            examples['label'].append(token_false)
    return examples


def split_pairs(pairs, seq2seq=True):
    if seq2seq:
        examples = {
            'label': [],
            'text': [],
        }
        for i in range(len(pairs['query'])):
            examples['text'].append(prompt_template.format(query=pairs["query"][i], text=pairs["passage"][i]))
            examples['label'].append([token_false, token_true][int(pairs["label"][i])])
    else:
        examples = {
            'label': [],
            'query': [],
            'text': [],
        }
        for i in range(len(pairs['query'])):
            examples['query'].append(pairs["query"][i])
            examples['text'].append(pairs["passage"][i])
            examples['label'].append([token_false, token_true][int(pairs["label"][i])])
    return examples


def tokenize(batch, seq2seq=True):
    if seq2seq:
        kwargs = {
            'text': batch['text'],
            'truncation': True,
        }
    else:
        kwargs = {
            'text': batch['query'],
            'text_pair': batch['text'],
            'truncation': 'only_second',
        }

    tokenized = tokenizer(
        **kwargs,
        padding=True,
        max_length=args.max_doc_length,
    )

    if seq2seq:
        tokenized['labels'] = tokenizer(batch['label'])['input_ids'] if seq2seq else batch['label']
    else:
        tokenized['labels'] = [[float(i)] for i in batch['label']]

    return tokenized


if __name__ == "__main__":
    parser = HfArgumentParser((Seq2SeqTrainingArguments, ExtraArguments))
    training_args, args = parser.parse_args_into_dataclasses()
    training_args.evaluation_strategy = "no"
    training_args.do_eval = False
    set_seed(training_args.seed)

    total_examples = None
    if training_args.max_steps > 0:
        total_examples = (
            training_args.gradient_accumulation_steps
            * training_args.per_device_train_batch_size
            * training_args.max_steps
            * torch.cuda.device_count()
        )

    config = AutoConfig.from_pretrained(args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    seq2seq = 'ForConditionalGeneration' in config.architectures

    if seq2seq:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)
        trainer_cls = Seq2SeqTrainer
        data_collator = DataCollatorForSeq2Seq(tokenizer)

        if 'flan' in args.base_model:
            prompt_template = flan_prompt
            token_false, token_true = flant5_outputs
        else:
            prompt_template = monot5_prompt
            token_false, token_true = monot5_outputs
    else:
        config.num_labels = 1
        config.problem_type = 'multi_label_classification'
        model = AutoModelForSequenceClassification.from_pretrained(
            args.base_model,
            config=config,
        )
        trainer_cls = Trainer
        data_collator = DataCollatorWithPadding(tokenizer)
        token_false, token_true = [0, 1]


    if args.triples:
        dataset = load_dataset(
            'csv',
            data_files=args.triples,
            sep='\t',
            names=('query', 'positive', 'negative'),
        )
        dataset = dataset.map(
            partial(split_triples, seq2seq=seq2seq),
            remove_columns=('query', 'positive', 'negative'),
            batched=True,
        )
    elif args.pairs:
        dataset = load_dataset(
            'csv',
            data_files=args.pairs,
            sep='\t',
            names=('query', 'passage', 'label'),
        )
        dataset = dataset.map(
            partial(split_pairs, seq2seq=seq2seq),
            remove_columns=('query', 'passage', 'passage'),
            batched=True,
        )
    else:
        raise Exception('We must define a triples or a pairs file.')

    if total_examples:
        dataset['train'] = dataset['train'].shuffle().select(range(total_examples))

    dataset = dataset.map(
        partial(tokenize, seq2seq=seq2seq),
        remove_columns=('text', 'label'),
        batched=True,
        desc='Tokenizing',
    )

    trainer = trainer_cls(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset['train'],
        data_collator=data_collator,
    )
    train_metrics = trainer.train()
    trainer.save_model(training_args.output_dir)
    trainer.save_state()
    trainer.save_metrics('train', train_metrics.metrics)
