import os
import ftfy
import yaml
import random

with open(f'{os.path.dirname(__file__)}/prompts/templates.yaml') as f:
    templates = yaml.safe_load(f)


class Prompt:
    def __init__(
            self,
            template=None,
            examples=None,
            tokenizer=None,
            max_doc_length=None,
            max_query_length=None,
            max_prompt_length=None,
            max_new_token=16,
    ):
        self.template = template
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_doc_length = max_doc_length
        self.max_query_length = max_query_length
        self.max_prompt_length = max_prompt_length
        self.max_new_token = max_new_token

    @classmethod
    def load(cls, name, *args, **kwargs):
        if name in templates:
            template = templates[name]
            prompt_class = {
                'dynamic': DynamicPrompt,
                'static': StaticPrompt,
            }[template['mode']]
            return prompt_class(template=template['template'], *args, **kwargs)
        else:
            if not os.path.exists(name):
                raise FileNotFoundError(f'Prompt file {name} was not found!')

            with open(name) as f:
                return StaticPrompt(template=f.read(), *args, **kwargs)
    
    def _truncate_max_doc_length(self, document):
        if self.max_doc_length:
            document = self.tokenizer.decode(
                self.tokenizer(document, truncation=True, max_length=self.max_doc_length)[
                    "input_ids"
                ]
            )
        return document


class StaticPrompt(Prompt):
    def build(self, document, *args, **kwargs):
        document = self._truncate_max_doc_length(document)

        prompt = self.template.format(document=document, query="").rstrip()

        if self.max_prompt_length:
            prompt_length = len(self.tokenizer.tokenize(prompt))
            if prompt_length + self.max_new_token > self.max_prompt_length:
                raise Exception(
                    f"Overflowing prompt (prompt length: {prompt_length} + {self.max_new_token}, \
                     max length: {self.max_prompt_length})"
                )

        return prompt


class DynamicPrompt(Prompt):
    def build(self, document, n_examples=3):
        random_examples = random.sample(self.examples, n_examples)

        prompt = ""
        for i in range(n_examples):
            _, _, query, doc = random_examples[i]
            query = self._truncate_max_query_length(query)
            doc = self._truncate_max_doc_length(doc)

            prompt += self.template.format(document=doc, query=query)

        document = ftfy.fix_text(document)
        if self.max_doc_length:
            document = self.tokenizer.decode(
                self.tokenizer(document, truncation=True, max_length=self.max_doc_length)[
                    "input_ids"
                ]
            )

        prompt += self.template.format(document=document, query="").rstrip()

        if self.max_prompt_length:
            prompt_length = len(self.tokenizer.tokenize(prompt))
            if prompt_length + self.max_new_token > self.max_prompt_length:
                raise Exception(
                    f"Overflowing prompt (prompt length: {prompt_length} + {self.max_new_token}, \
                     max length: {self.max_prompt_length})"
                )

        return prompt
    
    def _truncate_max_query_length(self, query):
        if self.max_query_length:
                query = self.tokenizer.decode(
                    self.tokenizer(query, truncation=True, max_length=self.max_query_length)[
                        "input_ids"
                    ]
                )
        return query
