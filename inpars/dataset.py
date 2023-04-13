import ftfy
import json
import random
import pandas as pd
from tqdm.auto import tqdm

def load_corpus(dataset_name, source='ir_datasets'):
    texts = []
    docs_ids = []

    if source == 'ir_datasets':
        import ir_datasets

        identifier = f'beir/{dataset_name}'
        if identifier in ir_datasets.registry._registered:
            dataset = ir_datasets.load(identifier)
        else:
            dataset = ir_datasets.load(dataset_name)

        for doc in tqdm(
            dataset.docs_iter(), total=dataset.docs_count(), desc="Loading documents from ir-datasets"
        ):
            texts.append(
                ftfy.fix_text(
                    f"{doc.title} {doc.text}"
                    if "title" in dataset.docs_cls()._fields
                    else doc.text
                )
            )
            docs_ids.append(doc.doc_id)
    else:
        from pyserini.search.lucene import LuceneSearcher
        from pyserini.prebuilt_index_info import TF_INDEX_INFO

        identifier = f'beir-v1.0.0-{dataset_name}.flat'
        if identifier in TF_INDEX_INFO:
            dataset = LuceneSearcher.from_prebuilt_index(identifier)
        else:
            dataset = LuceneSearcher.from_prebuilt_index(dataset_name)

        for idx in tqdm(range(dataset.num_docs), desc="Loading documents from Pyserini"):
            doc = json.loads(dataset.doc(idx).raw())
            texts.append(
                ftfy.fix_text(
                    f"{doc['title']} {doc['text']}"
                    if doc['title']
                    else doc['text']
                )
            )
            docs_ids.append(doc['_id'])

    return pd.DataFrame({'doc_id': docs_ids, 'text': texts})


def load_queries(dataset_name, source='ir_datasets'):
    queries = {}

    if source == 'ir_datasets':
        import ir_datasets
        dataset = ir_datasets.load(f'beir/{dataset_name}')

        for query in dataset.queries_iter():
            queries[query.query_id] = ftfy.fix_text(query.text)
    else:
        from pyserini.search import get_topics

        for (qid, data) in get_topics(f'beir-v1.0.0-{dataset_name}-test').items():
            queries[str(qid)] = ftfy.fix_text(data["title"])  # assume 'title' is the query

    return queries
