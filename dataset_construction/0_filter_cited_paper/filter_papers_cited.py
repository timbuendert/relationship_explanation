import json
import os
import subprocess
import gzip
import argparse
import pickle

METADATA_INPUT_DIR = 'metadata/'
PDF_PARSES_INPUT_DIR = 'pdf_parses/'

os.makedirs(METADATA_INPUT_DIR, exist_ok=True)
os.makedirs(PDF_PARSES_INPUT_DIR, exist_ok=True)

# shards
with open('../CS_dataset/S2ORC/final_dict.pkl', 'rb') as f:
    download_linkss = pickle.load(f)
    
# batches of papers
batches = [{
    'input_metadata_url': download_links['metadata'],
    'input_metadata_path': os.path.join(METADATA_INPUT_DIR,
                                        os.path.basename(download_links['metadata'].split('?')[0])),
    'input_pdf_parses_url': download_links['pdf_parses'],
    'input_pdf_parses_path': os.path.join(PDF_PARSES_INPUT_DIR,
                                          os.path.basename(download_links['pdf_parses'].split('?')[0])),
} for download_links in download_linkss]

parser = argparse.ArgumentParser()
parser.add_argument("--batches-start", default=None, type=int, required=True)
parser.add_argument("--batches-end", default=None, type=int, required=True)
parser.add_argument("--n", default=None, type=int, required=True)
args = parser.parse_args()

batches = batches[args.batches_start : args.batches_end]
print(f"{len(batches)} batch(es) considered (from {args.batches_start} to {args.batches_end})")

# load all IDs
all_ref_paper_ids = []
for i in range(10):    
    with open(f'../related_works/all_ref_paper_ids{i}.pkl', 'rb') as f:
        ids = pickle.load(f)
        all_ref_paper_ids += list(filter(None, ids))
all_ref_paper_ids = list(set(all_ref_paper_ids))
all_ref_paper_ids = list(map(int, all_ref_paper_ids))
print(f'Loaded in total {len(all_ref_paper_ids)} cited paper IDs!')

# iterate through all batches
for i, batch in enumerate(batches):
    print('\nBatch {}:'.format(i))

    cmd = ["wget", "-O", batch['input_pdf_parses_path'], batch['input_pdf_parses_url']]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    print('Download PDFs completed.')

    cmd = ["wget", "-O", batch['input_metadata_path'], batch['input_metadata_url']]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    print('Download Metadata completed.')

    with open("output/metadata_full_cited"+str(args.n)+".jsonl", "a") as acl_meta, gzip.open(batch['input_metadata_path'], 'rb') as f_meta:
        for n, line in enumerate(f_meta):
            if n % 10000 == 0:
                print(n)
            metadata_dict = json.loads(line)
            if int(metadata_dict['paper_id']) in all_ref_paper_ids:
                json.dump(metadata_dict, acl_meta)
                acl_meta.write("\n")

    # create a lookup for the pdf parse based on paper ID
    with open("output/pdf_parses_full_cited"+str(args.n)+".jsonl", "a") as acl_pdf, gzip.open(batch['input_pdf_parses_path'], 'rb') as f_pdf:
        for n, line in enumerate(f_pdf):
            if n % 10000 == 0:
                print(n)
            pdf_parse_dict = json.loads(line)
            if int(pdf_parse_dict['paper_id']) in all_ref_paper_ids:
                json.dump(pdf_parse_dict, acl_pdf)
                acl_pdf.write("\n")

    os.remove(batch['input_metadata_path'])
    os.remove(batch['input_pdf_parses_path'])