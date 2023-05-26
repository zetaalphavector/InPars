import torch
from datasets import load_dataset
from dataclasses import dataclass, field
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
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
        metadata={"help": "Triples file containing query, positive and negative examples (TSV format)."},
    )
    pairs: str = field(
        default=None,
        metadata={"help": "File containing pairs of query, passage and label (positive/negative) in TSV format."},
    )
    base_model: str = field(
        default="t5-base",
        metadata={"help": "Base model to fine-tune."},
    )
    max_doc_length: int = field(
        default=300,
        metadata={"help": "Maximum document length. Documents exceding this length will be truncated."},
    )

def split_triples(triples):
    examples = {
        'label': [],
        'text': [],
    }
    for i in range(len(triples['query'])):
        examples['text'].append(prompt_template.format(query=triples["query"][i], text=triples["positive"][i]))
        examples['label'].append(token_true)
        examples['text'].append(prompt_template.format(query=triples["query"][i], text=triples["negative"][i]))
        examples['label'].append(token_false)
    return examples

def split_pairs(pairs):
    examples = {
        'label': [],
        'text': [],
    }
    for i in range(len(pairs['query'])):
        examples['text'].append(prompt_template.format(query=pairs["query"][i], text=pairs["passage"][i]))
        examples['label'].append([token_false, token_true][int(pairs["label"][i])])
    return examples

def tokenize(batch):
    tokenized = tokenizer(
        batch['text'],
        padding=True,
        truncation=True,
        max_length=args.max_doc_length,
    )
    tokenized["labels"] = tokenizer(batch["label"])['input_ids']
    return tokenized


if __name__ == "__main__":
    parser = HfArgumentParser((Seq2SeqTrainingArguments, ExtraArguments))
    training_args, args = parser.parse_args_into_dataclasses()
    training_args.evaluation_strategy = "no"
    training_args.do_eval = False
    set_seed(training_args.seed)

    total_examples = None
    if training_args.max_steps > 0:
        total_examples = training_args.gradient_accumulation_steps * training_args.per_device_train_batch_size * training_args.max_steps * torch.cuda.device_count()

    model = AutoModelForSeq2SeqLM.from_pretrained(args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    if 'flan' in args.base_model:
        prompt_template = flan_prompt
        token_false, token_true = flant5_outputs
    else:
        prompt_template = monot5_prompt
        token_false, token_true = monot5_outputs

    if args.triples:
        dataset = load_dataset(
            'csv',
            data_files=args.triples,
            sep='\t',
            names=('query', 'positive', 'negative'),
        )
        dataset = dataset.map(
            split_triples,
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
            split_pairs,
            remove_columns=('query', 'passage', 'passage'),
            batched=True,
        )
    else:
        raise Exception('We must define a triples or a pairs file.')

    if total_examples:
        dataset['train'] = dataset['train'].shuffle().select(range(total_examples))
    dataset = dataset.map(
        tokenize,
        remove_columns=('text', 'label'),
        batched=True,
        desc='Tokenizing',
    )

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset['train'],
        data_collator=DataCollatorForSeq2Seq(tokenizer),
    )
    train_metrics = trainer.train()
    trainer.save_model(training_args.output_dir)
    trainer.save_state()
    trainer.save_metrics('train', train_metrics.metrics)
