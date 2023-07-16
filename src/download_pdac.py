#!/usr/bin/env python3

# %%
import glob
import json
import os

import pandas as pd

# %%
PDAC_DIR = "./data/pdac"
SERIES_ACCESSION = "GSE111672"
DATA_DIR = os.path.join(PDAC_DIR, SERIES_ACCESSION)
RAW_PATH = os.path.join(DATA_DIR, "suppl/GSE111672_RAW.tar")

os.system(
    f"wget -r -nH --cut-dirs=3 --no-parent 'ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE111nnn/{SERIES_ACCESSION}/' -P '{PDAC_DIR}'"
)

os.system(f"gunzip -r '{DATA_DIR}' -f")
os.system(f"tar -xf '{RAW_PATH}' --one-top-level -C '{DATA_DIR}'")
os.remove(RAW_PATH)
os.chdir(DATA_DIR)

# %%
# if not os.path.isfile("matrix/GSE111672_series_matrix.txt"):
#     os.system("wget ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE111nnn/GSE111672/matrix/GSE111672_series_matrix.txt.gz -P matrix/")
#     os.system("gunzip matrix/GSE111672_series_matrix.txt.gz")

# %%
with open("matrix/GSE111672_series_matrix.txt", "r") as f:
    lines = f.readlines()
lines = pd.Series(lines)

# %%
lines = lines.str.split("\t")
lines.index = lines.apply(lambda x: x[0])
lines = lines.apply(lambda x: x[1:])

# items = lines.explode()

# %%
# items = items.dropna()
# urls = items[items.map(lambda x: "ftp" in x and "samples" in x)]

# %%
# urls = urls.str.split('"').map(lambda x: x[1]).tolist()

# %%


# %%
accession_to_title = pd.DataFrame(
    [lines.loc["!Sample_title"], lines.loc["!Sample_geo_accession"]]
).T.applymap(lambda x: x.split('"')[1])
accession_to_title = accession_to_title.set_index(1)
accession_to_title = accession_to_title[0].to_dict()

with open("accession_to_title.json", "w") as f:
    json.dump(accession_to_title, f)

# %%


# %%
# urls_set = set(urls)

# for accession, title in accession_to_title.items():
#     if "indrop" in title.lower():
#         domain_dir = "indrop"
#     else:
#         domain_dir = "st"

#     out_dir = os.path.join(domain_dir, accession)
#     if not os.path.isdir(out_dir):
#         os.makedirs(out_dir)

#     accession_urls = [url for url in urls_set if accession in url]
#     for url in accession_urls:
#         os.system(f"wget {url} -P {out_dir}")
#         os.system(f"gunzip {out_dir}/*.gz")
#         urls_set.remove(url)

file_names = [os.path.basename(path) for path in glob.glob("GSE111672_RAW/*.gz")]
for file_name in file_names:
    print(file_name)

# %%

for accession, title in accession_to_title.items():
    if "indrop" in title.lower():
        domain_dir = "indrop"
    else:
        domain_dir = "st"

    out_dir = os.path.join(domain_dir, accession)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    accession_fnames = [fname for fname in file_names if accession in fname]
    for fname in accession_fnames:
        os.rename(f"GSE111672_RAW/{fname}", f"{out_dir}/{fname}")
        os.system(f"gunzip {out_dir}/{fname} -f")
        file_names.remove(fname)


# %%
os.rmdir("GSE111672_RAW")

# %%
