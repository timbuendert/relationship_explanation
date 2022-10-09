import random
import argparse
import pandas as pd

argparser = argparse.ArgumentParser()
argparser.add_argument('--n_samples', type=int)
argparser.add_argument('--seed', type=int)
args = argparser.parse_args()

# fix random sampling
random.seed(args.seed)

ind_outputs = [random.randint(0, 2475) for _ in range(0, args.n_samples)]
#print(dict(zip(list(range(1, len(ind_outputs) +1)), ind_outputs)))

# load all contents of test set
contents = pd.read_json(path_or_buf=f'../final_dataset/data/reflection_test.jsonl', lines=True)

output_string, output_string_solution = [], []

# directly add LaTeX code
output_string.append(
'''
\\documentclass[12pt,twoside,a4paper]{article}
\\usepackage[T1]{fontenc}
\\usepackage[utf8]{inputenc}
\\usepackage{enumitem}
\\setlist[enumerate,1]{labelindent=0pt, leftmargin=*}
\\usepackage{array,tabularx}
\\usepackage[a4paper,margin=1in,landscape]{geometry}
\\usepackage{lscape}

\\newcolumntype{P}{>{\\centering\\arraybackslash}p{0.75cm}}
\\newcolumntype{L}{>{\\raggedright\\arraybackslash}m{0.2\\textwidth}}
\\newcolumntype{R}{>{\\raggedleft\\arraybackslash}m{0.2\\textwidth}}

\\newcommand{\\printtblhdr}{%
\\textit{Other} \\hspace{2.7cm} \\textit{Relationship}  \\hspace{0.1cm} \\parbox[c]{1cm}{\\centering \\tiny{Not\\\\confident}} \\small \\hspace{1.3cm} \\textit{Vague} \\hspace{3.6cm} \\textit{Specific} \\hspace{0.2cm} \\parbox[c]{1cm}{\\centering \\tiny{Not\\\\confident}} \\small \\hspace{1.4cm} \\textit{Incorrect} \\hspace{2.9cm} \\textit{Correct} \\hspace{0.1cm} \\parbox[c]{1cm}{\\centering \\tiny{Not\\\\confident}}
}

\\newcommand{\\usetbl}{%
  \\begin{tabular}{@{}|*5{P|}@{}}
    \\hline
    1 & 2 & 3 & 4 & 5 \\\\
    \\hline
  \\end{tabular}
}


\\newcommand\\prop[1]{%
  \\noindent\\parbox[t]{25cm}{#1}%
  \\vspace{0.5cm}
  \\\\
  \\vspace{0.3cm}
  \\parbox[t]{8cm}{\\usetbl}%
  \\kern-4em
  \\makebox[0pt][l]{$\\square$}{}
  \\qquad
  \\qquad
  \\qquad  
  \\parbox[t]{8cm}{\\usetbl}%
  \\kern-3.5em
  \\makebox[0pt][l]{$\\square$}{}  
  \\qquad
  \\qquad
  \\qquad
  \\parbox[t]{8cm}{\\usetbl}%
  \\kern-3.5em
  \\makebox[0pt][l]{$\\square$}{} 
  \\\\
  \\printtblhdr
  \\\\
  \\noindent\\rule[0.5ex]{\\linewidth}{0.75pt}
}

\\begin{document}
\\section{Qualitative Analysis}
This document presents an essential part of the master thesis \\textit{"Explaining Relationships Between Academic Documents Using Generative Transformer Models"}. As the name suggests, it is aimed in this work to fine-tune pre-trained Transformer language models to deduce the relationship between two academic papers. To do so, several input representations of the two papers have been proposed, with a novel one being presented in the mentioned thesis. \\\\

\\noindent Since it is difficult to evaluate the generated explanations quantitatively, this study is conducted to gauge the quality of the generated relationship explanations using human judgement. A preliminary study focused on the comparisons between the varying Transformer models where the recently published \\textit{OPT} (Zhang et al., 2022) model stood out. Now, this study aims at judging the varying the context representations used for fine-tuning the \\textit{OPT} model. To this end, 30 randomly drawn samples from the test set will be presented along with three generated explanations (corresponding to three contexts representations). The order of these three explanations will be shuffled at each sample. Also, each sample will show the \\textit{target} which presents the in-line citation along with the principal (in which the citation appears) and cited (the paper that is cited) paper's titles and abstracts, to better evaluate the the explanations. Also, note that therefore the explanation is written naturally from the point of view of the principal paper. The \\texttt{\#REF} token is a placeholder for a citation of the corresponding paper. \\\\

\\noindent To judge the explanations in different dimensions, the three criteria \\textbf{relationship} (does the presented explanation describe a relationship or something else?), \\textbf{specificity} and \\textbf{correctness} must be scored on a Likert scale from 1 (lowest) to 5 (highest). In addition, there is an optional checkbox next to each rating as a way to indicate that this rating is not of high confidence, for example, because some content information from the papers might be missing. In turn, an unchecked box shows a high confidence in the given rating. \\\\

\\noindent \\subsection*{Thank you very much for taking part in this study, it is very much appreciated!}


\\newgeometry{left=1.5cm, bottom=1.5cm, top=1.5cm, right=1.5cm}
''')


output_string_solution.append(
'''
\\documentclass[12pt,twoside,a4paper]{article}
\\usepackage[T1]{fontenc}
\\usepackage[utf8]{inputenc}
\\usepackage{enumitem}
\\setlist[enumerate,1]{labelindent=0pt, leftmargin=*}
\\usepackage{array,tabularx}
\\usepackage[a4paper,margin=1in,landscape]{geometry}
\\usepackage{lscape}

\\newcolumntype{P}{>{\\centering\\arraybackslash}p{0.75cm}}
\\newcolumntype{L}{>{\\raggedright\\arraybackslash}m{0.2\\textwidth}}
\\newcolumntype{R}{>{\\raggedleft\\arraybackslash}m{0.2\\textwidth}}

\\newcommand{\\printtblhdr}{%
\\textit{Other} \\hspace{2.7cm} \\textit{Relationship}  \\hspace{0.1cm} \\parbox[c]{1cm}{\\centering \\tiny{Not\\\\confident}} \\small \\hspace{1.3cm} \\textit{Vague} \\hspace{3.6cm} \\textit{Specific} \\hspace{0.2cm} \\parbox[c]{1cm}{\\centering \\tiny{Not\\\\confident}} \\small \\hspace{1.4cm} \\textit{Incorrect} \\hspace{2.9cm} \\textit{Correct} \\hspace{0.1cm} \\parbox[c]{1cm}{\\centering \\tiny{Not\\\\confident}}\\\\
}

\\newcommand{\\usetbl}{%
  \\begin{tabular}{@{}|*5{P|}@{}}
    \\hline
    1 & 2 & 3 & 4 & 5 \\\\
    \\hline
  \\end{tabular}
}


\\newcommand\\prop[1]{%
  \\noindent\\parbox[t]{25cm}{#1}%
  \\vspace{0.5cm}
  \\\\
  \\vspace{0.3cm}
  \\parbox[t]{8cm}{\\usetbl}%
  \\qquad
  \\parbox[t]{8cm}{\\usetbl}%
  \\qquad
  \\parbox[t]{8cm}{\\usetbl}%
  \\\\
  \\printtblhdr
  \\\\
  \\noindent\\rule[0.5ex]{\\linewidth}{0.75pt}
}

\\begin{document}
\\section{Qualitative Analysis}
This document presents an essential part of the master thesis \\textit{"Explaining Relationships Between Academic Documents Using Generative Transformer Models"}. As the name suggests, it is aimed in this work to fine-tune pre-trained Transformer language models to deduce the relationship between two academic papers. To do so, several input representations of the two papers have been proposed, with a novel one being presented in the mentioned thesis. \\\\

\\noindent Since it is difficult to evaluate the generated explanations quantitatively, this study is conducted to gauge the quality of the generated relationship explanations using human judgement. A preliminary study focused on the comparisons between the varying Transformer models where the recently published \\textit{OPT} model (Zhang et al., 2022) stood out. Now, this study aims at judging the varying context representations used for fine-tuning the \\textit{OPT} model. To this end, 30 randomly drawn samples from the test set will be presented along with three generated explanations (corresponding to three contexts representations). The order of these three explanations will be shuffled at each sample. Also, each sample will show the \\textit{target} which presents the in-line citation along with the principal (paper in which the citation appears) and cited (the paper that is cited) paper's titles and abstracts, to better evaluate the the explanations. Also, note that therefore the explanation is naturally written from the point of view of the principal paper. The \\texttt{\#REF} token is a placeholder for a citation of the corresponding paper. \\\\

\\noindent To judge the explanations in different dimensions, the three criteria \\textbf{relationship} (does the presented explanation describe a relationship or something else?), \\textbf{specificity} and \\textbf{correctness} must be scored on a Likert scale from 1 (lowest) to 5 (highest). In addition, there is an optional checkbox next to each rating as a way to indicate that this rating is of low confidence, for example, because some content information from the papers might be missing. In turn, an unchecked box shows a high confidence in the given rating. \\\\

\\noindent \\subsection*{Thank you very much for taking part in this study, it is very much appreciated!}


\\newgeometry{left=1.5cm, bottom=1.5cm, top=1.5cm, right=1.5cm}
''')

# evaluate all three context representations
models = {'OPT - Cond. Sum.': f"OPT/outputs/eval_reflection_cond_sum_1_5.outputs",
          'OPT - Title-Abs': f"OPT/outputs/eval_reflection_title_abs.outputs",
          'OPT - Intro-Entity': f"OPT/outputs/eval_reflection_intro_entity.outputs",
          }


# retrieve samples
for n, ind in enumerate(ind_outputs):
    outputs = []
    with open(f"../contexts_reflection/cond_sum/test.target", "r") as tgt:
        n_counter = 0
        for line in tgt:
            if n_counter == ind:
                if '#' in line:
                    line = line.replace('#', '\#').replace('_', '\_').replace('%', '\%').replace('$', '\$')
                tgt = line
            n_counter += 1

    for m_n, model in enumerate(models): 
        path = models[model]

        with open(path, "r") as data:
            n_counter = 0
            for line in data:
                if n_counter == ind:
                    if '#' in line:
                        line = line.replace('#', '\#').replace('_', '\_').replace('%', '\%').replace('$', '\$')
                    outputs.append(line)
                n_counter += 1


    # get title; abstracts; texts of principal and cited documents
    sample_contents = contents.iloc[ind, :]

    p_title = sample_contents['principal_title'].replace('#', '\#').replace('_', '\_').replace('%', '\%').replace('$', '\$')
    c_title = sample_contents['cited_title'].replace('#', '\#').replace('_', '\_').replace('%', '\%').replace('$', '\$')
    p_abs = ' '.join([sample_contents['principal_abstracts'][i]['text'] for i in range(len(sample_contents['principal_abstracts']))]).replace('#', '\#').replace('_', '\_').replace('%', '\%').replace('$', '\$')
    c_abs = ''.join([sample_contents['cited_abstract'][i]['text'] for i in range(len(sample_contents['cited_abstract']))]).replace('#', '\#').replace('_', '\_').replace('%', '\%').replace('$', '\$')

    # could integrate entire text of principal and cited paper:
    #c_text = ' '.join([sample_contents['cited_text'][i]['text'] for i in range(len(sample_contents['cited_text']))])
    #p_text = ' '.join([sample_contents['principal_text'][i]['text'] for i in range(len(sample_contents['principal_text']))])

    order_outputs = list(range(len(models)))
    random.shuffle(order_outputs)

    output_string.append(f'''
\\newpage
\\subsection*{{Sample {n+1}: Texts}}
\\small{{
\\begin{{tabularx}}{{\linewidth}}{{X X}}
\\multicolumn{{1}}{{c}}{{\\textbf{{Principal Document}}}} & \multicolumn{{1}}{{c}}{{\\textbf{{Cited Document}}}} \\\\
\\textit{{{p_title}}} & \\textit{{{c_title}}} \\\\
{p_abs} & {c_abs} \\\\
\\end{{tabularx}}

\\newpage

\\subsection*{{Sample {n+1}: Relationship Explanations}}
\\hspace{{0.5cm}}

\\prop{{{tgt} (\\textbf{{Target}})}}
    ''')

    output_string_solution.append(f'''
\\newpage
\\subsection*{{Sample {n+1}: Texts}}
\\small{{
\\begin{{tabularx}}{{\linewidth}}{{X X}}
\\multicolumn{{1}}{{c}}{{\\textbf{{Principal Document}}}} & \multicolumn{{1}}{{c}}{{\\textbf{{Cited Document}}}} \\\\
\\textit{{{p_title}}} & \\textit{{{c_title}}} \\\\
{p_abs} & {c_abs} \\\\
\\end{{tabularx}}

\\newpage

\\subsection*{{Sample {n+1}: Relationship Explanations}}
\\hspace{{0.5cm}}

{tgt} (\\textbf{{Target}}) 

    ''')


    for i in order_outputs:
        output_string.append(f'''           
\\prop{{{outputs[i]}}}
        ''')

        output_string_solution.append(f'''           
\\textbf{{{list(models.keys())[i]}}} \\hspace{{0.5cm}} {outputs[i]}
        ''')

    output_string.append(f'''
}}
        ''')

    output_string_solution.append(f'''
}}
        ''')


output_string.append(
'''
\\end{document}
''')

output_string_solution.append(
'''
\\end{document}
''')

# output survey
with open(f'qualitative_study.tex', 'w') as fout:
    for i in range(len(output_string)):
        fout.write(output_string[i])

# output survey with corresponding context representations
with open(f'qualitative_study_solution.tex', 'w') as fout:
    for i in range(len(output_string_solution)):
        fout.write(output_string_solution[i])