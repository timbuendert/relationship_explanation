import pandas as pd
from tqdm import tqdm
import json
import pickle
import argparse

def get_title(id):
    filtered_df = meta[meta["paper_id"] == id]
    assert filtered_df.shape[0] == 1, print(filtered_df)
    return str(filtered_df["title"])

parser = argparse.ArgumentParser()
parser.add_argument("--n", default=None, type=int, required=True)
args = parser.parse_args()

print(f'File {args.n} is processed:')

# load data
meta = pd.read_json(path_or_buf=f'../CS_papers/data/metadata_full{args.n}.jsonl', lines=True, dtype= {'paper_id': str})
pdf = pd.read_json(path_or_buf=f'../CS_papersdata/pdf_parses_full{args.n}.jsonl', lines=True, dtype= {'paper_id': str})


# Create related work sections
counter = 0

with open(f'related_work{args.n}.jsonl',"a+") as f_pdf:
    for i in tqdm(range(pdf.shape[0])):
        filtered_dict = {}
        pdf_parse_dict = pdf.iloc[i,:]
        filtered_dict["paper_id"] = str(pdf_parse_dict["paper_id"])
        filtered_dict["title"] = get_title(str(pdf_parse_dict["paper_id"]))
        filtered_dict["abstract"] = pdf_parse_dict["abstract"]
        filtered_dict["bib_entries"] = pdf_parse_dict["bib_entries"]
        filtered_dict["related_work"] = []
        for paragraph in pdf_parse_dict["body_text"]:
            if ("related" in paragraph["section"].lower()) or ('previous' in paragraph["section"].lower()):
                filtered_dict["related_work"].append(paragraph)
        if len(filtered_dict["related_work"]) > 0:
            counter += 1
            json.dump(filtered_dict,f_pdf)
            f_pdf.write("\n")
        
print(counter)


# retrieve all cited paper ids in the related work sections
all_ref_paper_ids = set([])
with open(f'related_work{args.n}.jsonl',"r") as f_pdf:
    for line in tqdm(f_pdf):
        related_work_dict = json.loads(line)
        ref_ids = []
        for paragraph in related_work_dict["related_work"]:
            for ref in paragraph["cite_spans"]:
                ref_ids.append(ref["ref_id"])
        
        for ref in ref_ids:
            try:
                all_ref_paper_ids.add(related_work_dict["bib_entries"][ref]["link"])
            except:
                pass

print('Number of ref ids:', len(all_ref_paper_ids))

with open(f'related_works/all_ref_paper_ids{args.n}.pkl', 'wb') as f:
    pickle.dump(list(all_ref_paper_ids), f)
