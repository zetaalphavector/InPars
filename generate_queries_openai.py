import argparse
import json
import os
import re
import openai
from tqdm import tqdm


def parse_good_bad(data):
    text = data['text']
    if 'bad question:' not in text.lower():
        return '', ''
    good_question, bad_question = re.split('bad question:', text, flags=re.IGNORECASE)
    good_question = good_question.strip()
    bad_question = bad_question.strip()

    return good_question, bad_question


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--collection', type=str)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--engine', type=str, default='curie')
    parser.add_argument('--prompt_template', type=str, default='prompts/vanilla_prompt.txt')
    parser.add_argument('--max_examples', type=int, default=100000,
                        help='Maximum number of documents to read from the collection.')
    parser.add_argument('--max_tokens', type=int, default=64, help='Max tokens to be generated.')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Sampling temperature. Zero means greedy decoding.')
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--min_doc_chars', type=int, default=0,
                        help='Minimum number of chars an input document must have.')
    parser.add_argument('--max_doc_chars', type=int, default=100000,
                        help='Maximum number of chars an input document must have.')
    parser.add_argument('--sleep_time', type=float, default=1.5, 
                        help='Time to wait between API calls, in seconds.')
    parser.add_argument('--good_bad', action='store_true',
                        help='The model should produce a good question followed by a bad question.')          
    parser.add_argument('--include_doc_probs', action='store_true',
                        help='Wheter or not to save the tokens probabilities produeced by the model.')  

    args = parser.parse_args()

    openai.api_key = os.getenv('OPENAI_API_KEY')
    with open(args.prompt_template) as f:
        template_text = f.read()

    num_examples_so_far = 0
    skip_doc_ids = set()
    if os.path.exists(args.output):
        for line in open(args.output):
            skip_doc_ids.add(json.loads(line.strip())['doc_id'])
        num_examples_so_far = len(skip_doc_ids)

    n_docs_skipped = 0
    with open(args.collection) as f:
        with open(f'{args.output}', 'a') as fout:
            progress_bar = tqdm(total=args.max_examples)
            progress_bar.n = num_examples_so_far
            for line_num, line in enumerate(f):

                if num_examples_so_far >= args.max_examples:
                    break

                if len(line.strip().split('\t')) != 2:
                    n_docs_skipped += 1
                    print(f'Skipping due to bad formatting. Skipped {n_docs_skipped} docs so far')
                    continue

                doc_id, doc_text = line.strip().split('\t')

                if doc_id in skip_doc_ids:
                    n_docs_skipped += 1
                    print(f'Skipping because already seen. Skipped {n_docs_skipped} docs so far')
                    continue

                if len(doc_text) < args.min_doc_chars:
                    n_docs_skipped += 1
                    print(f'Skipping due to min len. Skipped {n_docs_skipped} docs so far')
                    continue

                if len(doc_text) > args.max_doc_chars:
                    n_docs_skipped += 1
                    print(f'Skipping due to max len. Skipped {n_docs_skipped} docs so far')
                    continue

                prompt_text = template_text.format(document_text=doc_text)
                output = openai.Completion.create(
                    engine=args.engine,
                    prompt=prompt_text,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    stop='\n\n',
                    logprobs=1,
                    echo=args.include_doc_probs)['choices'][0]

                if args.good_bad:
                    question, bad_question = parse_good_bad(output)

                else:
                    question = output['text']
                    bad_question = ''

                index_start = 0
                index_end = 0

                try:
                    index_end += output['logprobs']["tokens"][index_end:].index('\n')
                except ValueError:
                    index_end = len(output['logprobs']["tokens"])

                log_probs = output['logprobs']["token_logprobs"][index_start:index_end]

                output_dict = {
                    'doc_id': doc_id,
                    'doc_text': doc_text,
                    'question': question,
                    'log_probs': log_probs,
                }
                if bad_question:
                    output_dict['bad_question'] = bad_question

                fout.write(json.dumps(output_dict) + '\n')
                if line_num & (line_num - 1) == 0 or line_num % 1000 == 0:
                    # LOG every power of 2 or 1000 steps.
                    print(f'Document: {doc_text}\nQuestion: {question}\n')
                
                num_examples_so_far += 1
                progress_bar.update(1)
                time.sleep(args.sleep_time)

    print('Done!')
