# IR Final Project (2025–2026) — Wikipedia Search Engine

A search engine over the **English Wikipedia** corpus, built for the Information Retrieval final project.
It supports multiple ranking signals (body TF-IDF cosine, title match, anchor match, PageRank, PageViews)
and exposes a Flask API compatible with the course staff frontend specification.

## What’s in this repo

### Core files
- `search_frontend.py`  
  Flask server that loads the inverted indices + auxiliary signals and exposes the required endpoints:
  `/search`, `/search_body`, `/search_title`, `/search_anchor`, `/get_pagerank`, `/get_pageview`.

- `inverted_index_gcp.py`  
  Staff-provided helper implementation for writing/reading an inverted index in fixed-size binary shards
  and serializing index metadata (df + posting locations). Used as reference / compatible format.

### Indexing / Data preparation
- `IR_proj_gcp.py` and `IR_proj_gcp.ipynb`  
  Index building pipeline (Spark / Dataproc-friendly):
  - Build **body** index (`postings_gcp_body`)
  - Build **title** index (`postings_gcp_title`)
  - Build **anchor** index (`postings_gcp_anchor`)
  - Compute **PageRank** (`pr.pkl`)
  - Compute **PageViews** (`pageviews.pkl`)
  - Build **id→title mapping** (`id_to_title.pkl`)

### Deployment helpers
- `startup_script_gcp.sh`  
  VM startup script: installs Python deps, creates a venv, and prepares the environment.

- `run_frontend_in_gcp.sh`  
  Command sequence to:
  1) reserve a static external IP  
  2) open firewall for port `8080`  
  3) create a Compute Engine VM with the startup script  
  4) SCP `search_frontend.py` to the VM  
  5) run the Flask server with `nohup`  

### Development helper
- `run_frontend_in_colab.ipynb`  
  Notebook to run the frontend in Colab while pointing to the index files in GCS.

---

## Ranking methods implemented

The engine supports the following signals (as required by the project spec):

1. **Body cosine similarity TF-IDF** (`/search_body`)  
2. **Title binary match** (`/search_title`)  
3. **Anchor binary match** (`/search_anchor`)  
4. **PageRank** (`/get_pagerank`)  
5. **PageViews** (`/get_pageview`)  
6. **Combined ranker** (`/search`)  
   Weighted combination of the available signals.  
   If some auxiliary data is not loaded (low-memory mode), weights are renormalized automatically.

---

## Endpoints

Base URL: `http://<HOST>:8080`

- `GET /search?query=<text>`  
  Returns top 100 results using the combined ranker.

- `GET /search_body?query=<text>`  
  Body TF-IDF cosine similarity.

- `GET /search_title?query=<text>`  
  Title match (binary count of query terms in titles).

- `GET /search_anchor?query=<text>`  
  Anchor match (binary count of query terms in anchor text).

- `POST /get_pagerank`  
  Body: JSON list of wiki_ids (ints)  
  Returns: JSON list of pagerank scores.

- `POST /get_pageview`  
  Body: JSON list of wiki_ids (ints)  
  Returns: JSON list of pageview counts.

All search endpoints return a JSON list of `(wiki_id, title)` pairs.

---

## Data / Index layout in GCS

This project expects the following objects to exist in your **Google Cloud Storage bucket**:

### Posting directories (binary shards + posting-locs pickles)
- `postings_gcp_body/`
- `postings_gcp_title/`
- `postings_gcp_anchor/`

Each folder contains:
- `<bucket_id>_XXX.bin` shards (posting lists)
- `<bucket_id>_posting_locs.pickle` (posting locations for that bucket)
- Index metadata pickle:
  - `index_body.pkl` (or fallback name `index.pkl`)
  - `index_title.pkl` (or fallback name `index.pkl`)
  - `index_anchor.pkl` (or fallback name `index.pkl`)

### Auxiliary pickles
- `pr.pkl` — dict: `{doc_id: pagerank}`
- `pageviews.pkl` — dict: `{doc_id: views}`
- `id_to_title.pkl` — dict: `{doc_id: title}`

---

## Configuration (environment variables)

`search_frontend.py` is configured primarily via environment variables so it can run in:
local dev, Colab, and GCP VM with minimal changes.

### Required (typically)
- `BUCKET_NAME` — GCS bucket containing the index + pickles  
- `PROJECT_ID` — GCP project (optional if default credentials cover it)

### Index folder names (defaults shown)
- `BODY_DIR=postings_gcp_body`
- `TITLE_DIR=postings_gcp_title`
- `ANCHOR_DIR=postings_gcp_anchor`

### Index pkl names (defaults shown)
- `BODY_INDEX_NAME=index_body`
- `TITLE_INDEX_NAME=index_title`
- `ANCHOR_INDEX_NAME=index_anchor`

### Auxiliary loading controls (important for small VMs)
To avoid OOM on small machines, the server can run in a “lazy/low-memory” style.

- `LOAD_TITLES=auto|1|0` (default: `auto`)
- `LOAD_PAGERANK=auto|1|0` (default: `auto`)
- `LOAD_PAGEVIEWS=auto|1|0` (default: `auto`)
- `LAZY_AUX=1|0` (default: `1`)
- `LOW_ME_
