import argparse
import random
import torch 
import numpy as np
from tqdm import tqdm

argparser = argparse.ArgumentParser()
argparser.add_argument('--model', type=str)
argparser.add_argument('--context', type=str)
argparser.add_argument('--n_samples', type=int)
argparser.add_argument('--seed', type=int)
args = argparser.parse_args()

random.seed(args.seed)


intent_lengths = {'single_summ': 64003, 'reflection': 2475} # number of test examples from which can be sampled
scrs = [] # [[] for _ in range(len(intent_lengths.keys()))]
targets = []
outputs = [[] for _ in range(len(intent_lengths.keys()))]

# output arguments
n_samples = 1
samples_length = 55
temperature = 1.0
top_k = 0
top_p = 0.9
repetition_penalty = 1.0
length_penalty = 2
no_repeat_ngram = 3
num_beams = 4


def generate_output_scigpt2(model_path, samples_length):
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    from SciGen.scigpt2_generate import sample_sequence

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2',  truncation = True) 
    special_tokens = {"additional_special_tokens": ["<|tgt|>"], 'sep_token': '<|SEP|>', 'pad_token': '<|PAD|>'}
    if args.context == 'intro_tfidf':
        special_tokens['additional_special_tokens'].append('<TFIDF>')
    if args.context == 'intro_entity':
        special_tokens['additional_special_tokens'].append('<TFIDF>')
        special_tokens['additional_special_tokens'].append('<ENT>')
    tokenizer.add_special_tokens( special_tokens )

    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.to(device)
    model.eval()
    model.resize_token_embeddings(len(tokenizer))

    if samples_length < 0 and model.config.max_position_embeddings > 0:
        samples_length = model.config.max_position_embeddings
    elif 0 < model.config.max_position_embeddings < samples_length:
        samples_length = model.config.max_position_embeddings  # No generation bigger than model size 
    elif samples_length < 0:
        samples_length = int(10000) 

    output_texts = []
    for sample in tqdm(scrs):
        sample = sample.replace('\n', '')
        raw_text = sample.replace('<DOC_SEP>', '<|SEP|>') + ' <|tgt|>'
        context = tokenizer.tokenize(raw_text)
        context_tokens = tokenizer.convert_tokens_to_ids( context )

        sep_token_ind = tokenizer.convert_tokens_to_ids('<|SEP|>')
        if len(context_tokens) > 1024-(samples_length+3):
            n_remove_tokens = len(context_tokens) - (1024-(samples_length+3))
            ind_doc = context_tokens.index(sep_token_ind)
            len_princ = len(context_tokens[:ind_doc])
            len_cited = len(context_tokens[(ind_doc+1):])

            if (len_cited/(len_princ + len_cited)) > 0.75:
                remove_cited = n_remove_tokens
                remove_princ = 0

            elif (len_princ/(len_princ + len_cited)) > 0.75:
                remove_princ = n_remove_tokens
                remove_cited = 0

            else:
                remove_princ = int(np.ceil(n_remove_tokens * (len_princ/(len_princ + len_cited))))
                remove_cited = int(np.ceil(n_remove_tokens * (len_cited/(len_princ + len_cited))))

            context_tokens = context_tokens[:(ind_doc-remove_princ)] + context_tokens[(ind_doc):-(1+remove_cited)] + [context_tokens[-1]]

            
        out = sample_sequence(
            model=model,
            context=context_tokens,
            tokenizer = tokenizer,
            num_samples=n_samples,
            length=samples_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            device=device,
        )
        out = out[:, len(context_tokens):].tolist()


        for o in out:
            text = tokenizer.decode(o, clean_up_tokenization_spaces=True)
        output_texts.append(text.replace('#', '\#').replace('_', '\_').replace('%', '\%'))
    return output_texts


# get n_samples many context samples per intent
for n, intent in enumerate(intent_lengths.keys()):
    ind_samples = [random.randint(0, intent_lengths[intent]) for _ in range(0, args.n_samples)] #.sort()
    with open(f"../contexts_{intent}/{args.context}/test.source", "r") as src:
        n_counter = 0
        for line in src:
            if n_counter in ind_samples:
                #srcs[n].append(line)
                scrs.append(line)
            n_counter += 1

    with open(f"../contexts_{intent}/{args.context}/test.target", "r") as tgt:
        n_counter = 0
        for line in tgt:
            if n_counter in ind_samples:
                targets.append(str(line).replace('#', '\#').replace('_', '\_').replace('%', '\%'))
            n_counter += 1


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
batch_size = 4

# generate outputs for retrieved samples
for n, intent in tqdm(enumerate(intent_lengths.keys())):
    
    # load models and tokenizers
    if args.model == 'BART':
        path = f"BART/models/bart_{intent}_{args.context}"

#        from BART.transformers_src.utils import (use_task_specific_params)
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i: i + n]

        def use_task_specific_params(model, task):
            """Update config with summarization specific params."""
            task_specific_params = model.config.task_specific_params
            if task_specific_params is not None:
                pars = task_specific_params.get(task, {})
                model.config.update(pars)

        model = AutoModelForSeq2SeqLM.from_pretrained(path.replace('pytorch_model.bin', '')).to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(path.replace('pytorch_model.bin', ''))
        # add_special_tokens_(model, tokenizer)

        use_task_specific_params(model, 'summarization') # update config with task specific params
        prefix = None
        if prefix is None:
            prefix = prefix or getattr(model.config, "prefix", "") or ""
        for examples_chunk in tqdm(list(chunks(scrs, batch_size))):
            examples_chunk = [prefix + text for text in examples_chunk]
            batch = tokenizer(examples_chunk, return_tensors="pt",
                            truncation=True, padding="longest").to(device)
            summaries = model.generate(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                length_penalty = length_penalty, 
                no_repeat_ngram_size = no_repeat_ngram, 
                num_beams = num_beams
            )
            dec = tokenizer.batch_decode(
                summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for hypothesis in dec:
                outputs[n].append(hypothesis.replace('#', '\#').replace('_', '\_').replace('%', '\%'))
                # print(hypothesis + "\n")


    elif args.model == 'SciGPT2':
        path = f"SciGen/training/{intent}_{args.context}"
        outputs[n] += generate_output_scigpt2(path, samples_length)


    elif args.model == 'SciGen':
        path = f"SciGen/scigen"
        outputs[n] = generate_output_scigpt2(path, samples_length)


    elif args.model == 'OPT':
        path = f"OPT/training/{intent}_{args.context}"
        from transformers import GPT2Tokenizer, OPTForCausalLM
        from OPT.opt_generate import sample_sequence, filter_output

        tokenizer = GPT2Tokenizer.from_pretrained(path,  truncation = True) 
        special_tokens = {"additional_special_tokens": ["<|tgt|>"], 'sep_token': '<|SEP|>', 'pad_token': '<|PAD|>'}
        if args.context == 'intro_tfidf':
            special_tokens['additional_special_tokens'].append('<TFIDF>')
        if args.context == 'intro_entity':
            special_tokens['additional_special_tokens'].append('<TFIDF>')
            special_tokens['additional_special_tokens'].append('<ENT>')
        tokenizer.add_special_tokens( special_tokens )

        model = OPTForCausalLM.from_pretrained(path)
        model.to(device)
        model.eval()
        model.resize_token_embeddings(len(tokenizer))

        if samples_length < 0 and model.config.max_position_embeddings > 0:
            samples_length = model.config.max_position_embeddings
        elif 0 < model.config.max_position_embeddings < samples_length:
            samples_length = model.config.max_position_embeddings  # No generation bigger than model size 
        elif samples_length < 0:
            samples_length = int(10000) 

        for sample in tqdm(scrs):
            sample = '</s>' + sample
            sample = sample.replace('\n', '')
            raw_text = sample.replace('<DOC_SEP>', '<|SEP|>') + ' <|tgt|>'
            context = tokenizer.tokenize(raw_text)
            context_tokens = tokenizer.convert_tokens_to_ids( context )

            sep_token_ind = tokenizer.convert_tokens_to_ids('<|SEP|>')
            if len(context_tokens) > 1024-(samples_length+3):
                n_remove_tokens = len(context_tokens) - (1024-(samples_length+3))
                ind_doc = context_tokens.index(sep_token_ind)
                len_princ = len(context_tokens[:ind_doc])
                len_cited = len(context_tokens[(ind_doc+1):])

                if (len_cited/(len_princ + len_cited)) > 0.75:
                    remove_cited = n_remove_tokens
                    remove_princ = 0

                elif (len_princ/(len_princ + len_cited)) > 0.75:
                    remove_princ = n_remove_tokens
                    remove_cited = 0

                else:
                    remove_princ = int(np.ceil(n_remove_tokens * (len_princ/(len_princ + len_cited))))
                    remove_cited = int(np.ceil(n_remove_tokens * (len_cited/(len_princ + len_cited))))

                context_tokens = context_tokens[:(ind_doc-remove_princ)] + context_tokens[(ind_doc):-(1+remove_cited)] + [context_tokens[-1]]

            out = sample_sequence(
                model=model,
                context=context_tokens,
                num_samples=n_samples,
                length=samples_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                device=device,
            )
            out = out[:, len(context_tokens):].tolist()

            for o in out:
                text = tokenizer.decode(o, clean_up_tokenization_spaces=True)
                text = filter_output(text) #text[: text.find(args.stop_token) if args.stop_token else None]
            outputs[n].append(text.replace('#', '\#').replace('_', '\_').replace('%', '\%'))

    else:
        print('Please input a valid model type.')


output_string = []

samples_per_page = 3

for i in range(len(scrs)):
    if i % samples_per_page == 0:
        output_string.append(f'''\\begin{{tabularx}}{{\linewidth}}{{@{{}}>{{\\bfseries}}l@{{\hspace{{2em}}}}X@{{}}}}''')

    output_string.append(f'''\n\\textbf{{Sample {i+1}}} & \\\ ''')
    if i < args.n_samples: # first n_samples are from first intent
        intent = list(intent_lengths.keys())[0].capitalize().replace('_', '\\_')
    else:
        intent = list(intent_lengths.keys())[1].capitalize().replace('_', '\\_') # second half from other intent
    #print(f'Target ({intent.capitalize()}): {targets[i]}')
    #for n, intent in enumerate(intent_lengths.keys()):
        #print(f'{intent.capitalize()}: {outputs[n][i]}')
    output_string.append(
f'''
\small{{\\textbf{{Single\\_summ}}}} & \small{{{outputs[0][i]}}} \\\ 
\small{{\\textbf{{Reflection}}}} & \small{{{outputs[1][i]}}} \\\ 
\hline
\small{{\\textbf{{Target ({intent})}}}} & \small{{{targets[i]}}} \\\ 
\\vspace{{1cm}} \\\ \n
'''
    )

    if i % samples_per_page == (samples_per_page - 1):
        output_string.append(f'''\end{{tabularx}}\n''')
        output_string.append(f'''\\newpage\n''')

    elif i == len(scrs):
        output_string.append(f'''\end{{tabularx}}\n''')
        output_string.append(f'''\\newpage\n''')

with open(f'intent_comparison_{args.model}_{args.context}.tex', 'w') as fout:
    for i in range(len(output_string)):
        fout.write(output_string[i])
