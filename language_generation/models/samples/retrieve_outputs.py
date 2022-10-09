import random
import argparse
import pandas as pd

argparser = argparse.ArgumentParser()
argparser.add_argument('--n_samples', type=int)
argparser.add_argument('--seed', type=int)
argparser.add_argument('--model_comparisons', action='store_true')

args = argparser.parse_args()

# fix random sampling
random.seed(args.seed)

models = ['SciGen', 'SciGPT2', 'OPT', 'BART', 'PEGASUS']
contexts = ['cond_sum_1_5', 'title_abs', 'intro_entity']
ind_outputs = [random.randint(0, 2475) for _ in range(0, args.n_samples)]

contents = pd.read_json(path_or_buf=f'../final_dataset/data/reflection_test.jsonl', lines=True)

output_string = []

# go through all models and context combinations to retrieve samples
for m in models:
    for n_c, c in enumerate(contexts):
        c_text = c.replace('_', '\_')
        if m == 'BART':
            path = f"BART/experiments/bart_reflection_{c}/test.output"
        elif m == 'SciGPT2':
            path = f"SciGen/outputs/eval_reflection_{c}.outputs"
        elif m == 'SciGen':
            path = f"SciGen/outputs/scigen_reflection_{c}.outputs"
        elif m == 'OPT':
            path = f"OPT/outputs/eval_reflection_{c}.outputs"
        elif m == 'PEGASUS':
            path = f"PEGASUS/experiments/pegasus_reflection_{c}/test.output"

        outputs = list(open(path).readlines())
        sel_outputs = [outputs[i].replace('#', '\#').replace('_', '\_').replace('%', '\%') for i in ind_outputs]

        if n_c != 0:
            output_string.append('''
\\clearpage 
\\section{{Samples of {m}}}
            ''')

        output_string.append(f'''
\\section{{Samples of {m}}}
\\begin{{table}}[h!]
\\centering
\\footnotesize{{
\\begin{{tabularx}}{{\linewidth}}{{X}}
'''
        )

        for n in range(args.n_samples):
            if n < len(sel_outputs)-1:
                output_string.append(f'''
{sel_outputs[n]} \\\\
\\vspace{{0.1cm}}
''' 
             )
            
            else:
                output_string.append(f'''
{sel_outputs[n]}
'''
             )

        output_string.append(f'''
\\end{{tabularx}}
}}
\\caption{{Samples using \\textit{{{c_text}}} as context representation}}
\\end{{table}}
    ''')


with open(f'qualitative_samples_{args.n_samples}.tex', 'w') as fout:
    for i in range(len(output_string)):
        fout.write(output_string[i])
