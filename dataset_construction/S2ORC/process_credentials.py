import re
import pickle

metadata, pdfs = [], []
url_regex = re.compile("'.*'$")

# retrieve urls from credentials text file
with open('S2ORC/current_credentials.txt') as cred:
    for line in cred:
        if len(line) < 10:
            continue
        url = url_regex.search(line).group()[1:-1]
        if 'metadata' in url:
            metadata.append(url)
        elif 'pdf_parses' in url:
            pdfs.append(url)
        else:
            raise ValueError

print(f'Number of metadata URLs: {len(metadata)}')
print(f'Number of pdf-parses URLs: {len(pdfs)}')

final_dict = [{"metadata": metadata[i], "pdf_parses": pdfs[i]} for i in range(len(pdfs))]

# export
with open('S2ORC/final_dict.pkl', 'wb') as f:
    pickle.dump(final_dict, f)