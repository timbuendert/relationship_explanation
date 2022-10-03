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
with open('S2ORC/final_dict.pkl', 'rb') as f:
    download_linkss = pickle.load(f)
    
# turn these into batches of work
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

for i, batch in enumerate(batches):
    print('\nBatch {}:'.format(i))

    cmd = ["wget", "-O", batch['input_pdf_parses_path'], batch['input_pdf_parses_url']]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    print('Download PDFs completed.')

    cmd = ["wget", "-O", batch['input_metadata_path'], batch['input_metadata_url']]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    print('Download Metadata completed.')

    cited_paper_ids = {}

    # filter papers using metadata values
    filtered_paper_id = []
    with gzip.open(batch['input_metadata_path'], 'rb') as f_meta:
        for line in f_meta:
            metadata_dict = json.loads(line)
            paper_id = metadata_dict['paper_id']
            #print(f"Currently viewing S2ORC paper: {paper_id}")

            if not metadata_dict['mag_field_of_study']:
                metadata_dict['mag_field_of_study'] = []
                            
            check_cs = False
            for field in metadata_dict['mag_field_of_study']:
                if 'computer' in field.lower():
                    check_cs = True
            

            # only care about ACL anthology and CS papers
            if (not metadata_dict['acl_id']) and (not check_cs):
                continue

            # and we want only papers with resolved outbound citations
            if not metadata_dict['has_outbound_citations']:
                continue

            filtered_paper_id.append(paper_id)

    filtered_paper_id = set(filtered_paper_id)
    print('{} filtered paper IDs'.format(len(filtered_paper_id)))

    with open("data/metadata_full"+str(args.n)+".jsonl", "a") as acl_meta, gzip.open(batch['input_metadata_path'], 'rb') as f_meta:
        for line in f_meta:
            metadata_dict = json.loads(line)
            if metadata_dict['paper_id'] in filtered_paper_id:
                json.dump(metadata_dict, acl_meta)
                acl_meta.write("\n")

    # create a lookup for the pdf parse based on paper ID
    with open("data/pdf_parses_full"+str(args.n)+".jsonl", "a") as acl_pdf, gzip.open(batch['input_pdf_parses_path'], 'rb') as f_pdf:
        for line in f_pdf:
            pdf_parse_dict = json.loads(line)
            if pdf_parse_dict['paper_id'] in filtered_paper_id:
                json.dump(pdf_parse_dict, acl_pdf)
                acl_pdf.write("\n")

    os.remove(batch['input_metadata_path'])
    os.remove(batch['input_pdf_parses_path'])


# python filter_ACL_S2ORC.py --batches_start=1 --batches_end=1 --n=1
