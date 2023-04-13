import os
import ftfy
import torch
import random
import statistics
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import urllib.error
from .prompt import Prompt

def load_examples(corpus, n_fewshot_examples):
    try:
        df = pd.read_json(
            f'https://huggingface.co/datasets/inpars/fewshot-examples/resolve/main/data/{corpus}.json',
            lines=True
        )
        # TODO limitar numero de exemplos (?)
        df = df[['query_id', 'doc_id', 'query', 'document']].values.tolist()
        random_examples = random.sample(df, n_fewshot_examples)
        with open('query_ids_to_remove_from_eval.tsv', 'w') as fout:
            for item in random_examples:
                fout.write(f'{item[0]}\t{item[2]}\n')

        return random_examples
    except urllib.error.HTTPError:
        return []


class InPars:
    def __init__(
        self,
        base_model="EleutherAI/gpt-j-6B",
        revision=None,
        corpus="msmarco",
        prompt=None,
        n_fewshot_examples=None,
        max_doc_length=None,
        max_query_length=None,
        max_prompt_length=None,
        max_new_tokens=64,
        fp16=False,
        int8=False,
        device=None,
        tf=False,
        torch_compile=False,
        verbose=False,
    ):
        self.corpus = corpus
        self.max_doc_length = max_doc_length
        self.max_query_length = max_query_length
        self.max_prompt_length = max_prompt_length
        self.max_new_tokens = max_new_tokens
        self.n_fewshot_examples = n_fewshot_examples
        self.device = device
        self.verbose = verbose
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model, padding_side="left"
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {"revision": revision}
        if fp16:
            model_kwargs["torch_dtype"] = torch.float16
            model_kwargs["low_cpu_mem_usage"] = True
        if int8:
            model_kwargs["load_in_8bit"] = True
            model_kwargs["device_map"] = "auto"

        if fp16 and base_model == "EleutherAI/gpt-j-6B":
            model_kwargs["revision"] = "float16"

        self.tf = tf

        if self.tf:
            from transformers import TFAutoModelForCausalLM
            self.model = TFAutoModelForCausalLM.from_pretrained(
                base_model,
                revision=revision,
            )
            self.model.config.pad_token_id = self.model.config.eos_token_id
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model, **model_kwargs
            )
            if torch_compile:
                self.model = torch.compile(self.model)
            self.model.to(self.device)
            self.model.eval()

        self.fewshot_examples = load_examples(corpus, self.n_fewshot_examples)
        self.prompter = Prompt.load(
            name=prompt,
            examples=self.fewshot_examples,
            tokenizer=self.tokenizer,
            max_query_length=self.max_query_length,
            max_doc_length=self.max_doc_length,
            max_prompt_length=self.max_prompt_length,
            max_new_token=self.max_new_tokens,
        )


    @torch.no_grad()
    def generate(
        self,
        documents,
        doc_ids,
        batch_size=1,
        **generate_kwargs,
    ):
        if self.tf:
            import tensorflow as tf

        disable_pbar = False if len(documents) > 1_000 else True
        prompts = [
            self.prompter.build(document, n_examples=self.n_fewshot_examples)
            for document in tqdm(documents, disable=disable_pbar, desc="Building prompts")
        ]

        if self.tf:
            generate = tf.function(self.model.generate, jit_compile=True)
            padding_kwargs = {"pad_to_multiple_of": 8}
        else:
            generate = self.model.generate
            padding_kwargs = {}

        results = []
        for i in tqdm(range(0, len(prompts), batch_size), desc="Generating queries"):
            batch_prompts = prompts[i : i + batch_size]
            batch_docs = documents[i : i + batch_size]
            batch_doc_ids = doc_ids[i : i + batch_size]

            tokens = self.tokenizer(
                batch_prompts,
                padding=True,
                truncation=True,
                return_tensors="tf" if self.tf else "pt",
                **padding_kwargs,
            )

            if not self.tf:
                tokens.to(self.device)

            outputs = generate(
                input_ids=tokens["input_ids"].long(),
                attention_mask=tokens["attention_mask"].long(),
                max_new_tokens=self.max_new_tokens,
                output_scores=True,
                return_dict=True,
                return_dict_in_generate=True,
                eos_token_id=198,  # hardcoded Ċ id
                bos_token_id=self.tokenizer.bos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                **generate_kwargs,
            )

            preds = [
                sequence.strip()
                for sequence in self.tokenizer.batch_decode(
                    outputs.sequences[:, tokens["input_ids"].shape[-1]:],
                    skip_special_tokens=True,
                )
            ]

            # Greedy decoding
            gen_sequences = outputs.sequences[:, tokens["input_ids"].shape[-1]:]
            pad_mask = (
                outputs.sequences[:, tokens["input_ids"].shape[-1]:]
                == self.tokenizer.pad_token_id
            )
            probs = torch.stack(outputs.scores, dim=1).log_softmax(dim=-1)
            prob_values, prob_indices = probs.max(dim=2)

            sequence_scores = [prob_values[i][~pad_mask[i]].tolist()[:-1] for i in range(len(prob_values))]

            for (question, log_probs, prompt_text, doc_id, document) in zip(preds, sequence_scores, batch_prompts, batch_doc_ids, batch_docs):
                results.append(
                    { 
                        "query": question,
                        "log_probs": log_probs,
                        "prompt_text": prompt_text,
                        "doc_id": doc_id,
                        "doc_text": document,
                        "fewshot_examples": [example[0] for example in self.fewshot_examples],
                    }
                )

        return results
