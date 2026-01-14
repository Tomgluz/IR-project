from flask import Flask, request, jsonify
import os
import math
import re
import pickle
from pathlib import Path
from collections import Counter, defaultdict
from contextlib import closing

import nltk
from nltk.corpus import stopwords
from google.cloud import storage

# ----------------------------
# Configuration (edit if needed)
# ----------------------------
# Prefer environment variables so the same code works in Colab, local dev, and GCP VM.
PROJECT_ID = os.environ.get("PROJECT_ID", "irproject2026")
BUCKET_NAME = os.environ.get("BUCKET_NAME", "irprojectbucket")

# Index folders in the bucket / local filesystem (same as in your IR_proj_gcp.py)
BODY_DIR = os.environ.get("BODY_DIR", "postings_gcp_body")
TITLE_DIR = os.environ.get("TITLE_DIR", "postings_gcp_title")
ANCHOR_DIR = os.environ.get("ANCHOR_DIR", "postings_gcp_anchor")

# Index pkl names (your IR_proj_gcp.py writes: index_body.pkl, index_title.pkl, index_anchor.pkl)
BODY_INDEX_NAME = os.environ.get("BODY_INDEX_NAME", "index_body")
TITLE_INDEX_NAME = os.environ.get("TITLE_INDEX_NAME", "index_title")
ANCHOR_INDEX_NAME = os.environ.get("ANCHOR_INDEX_NAME", "index_anchor")

# ----------------------------
# Storage helpers
# ----------------------------
def _get_bucket():
    # On GCP VM / Dataproc, default credentials are available.
    # In Colab you might authenticate explicitly (as in the staff notebook).
    try:
        client = storage.Client(project=PROJECT_ID) if PROJECT_ID else storage.Client()
    except Exception:
        client = storage.Client()
    return client.bucket(BUCKET_NAME)

def _open(path: str, mode: str):
    """Open either a local path or a GCS blob path, depending on what exists.

    - Local: path exists on filesystem
    - GCS: otherwise open blob from BUCKET_NAME
    """
    if os.path.exists(path):
        return open(path, mode)
    # If local doesn't exist, try GCS.
    bucket = _get_bucket()
    blob = bucket.blob(path)
    return blob.open(mode)

# ----------------------------
# Inverted index + posting reader
# ----------------------------
BLOCK_SIZE = 1999998
TUPLE_SIZE = 6  # 4 bytes doc_id + 2 bytes tf

class MultiFileReader:
    """Sequential binary reader of multiple posting files (local or GCS)."""

    def __init__(self, base_dir: str):
        self._base_dir = base_dir
        self._open_files = {}

    def __enter__(self):
        return self

    def read(self, locs, n_bytes: int) -> bytes:
        b = []
        for f_name, offset in locs:
            full_path = str(Path(self._base_dir) / f_name)
            if full_path not in self._open_files:
                self._open_files[full_path] = _open(full_path, "rb")
            f = self._open_files[full_path]
            f.seek(offset)
            n_read = min(n_bytes, BLOCK_SIZE - offset)
            b.append(f.read(n_read))
            n_bytes -= n_read
        return b"".join(b)

    def close(self):
        for f in self._open_files.values():
            try:
                f.close()
            except Exception:
                pass
        self._open_files = {}

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


class InvertedIndex:
    def __init__(self):
        self.df = Counter()
        self.term_total = Counter()
        self.posting_locs = defaultdict(list)

    @staticmethod
    def read_index(base_dir: str, name: str):
        path = str(Path(base_dir) / f"{name}.pkl")
        with _open(path, "rb") as f:
            return pickle.load(f)


def read_posting_list(index: InvertedIndex, base_dir: str, term: str):
    """Return posting list [(doc_id, tf), ...] for a given term."""
    if term not in index.posting_locs:
        return []
    locs = index.posting_locs[term]
    with closing(MultiFileReader(base_dir)) as reader:
        b = reader.read(locs, index.df[term] * TUPLE_SIZE)

    posting_list = []
    for i in range(index.df[term]):
        doc_id = int.from_bytes(b[i * TUPLE_SIZE : i * TUPLE_SIZE + 4], "big")
        tf = int.from_bytes(b[i * TUPLE_SIZE + 4 : (i + 1) * TUPLE_SIZE], "big")
        posting_list.append((doc_id, tf))
    return posting_list

# ----------------------------
# Query processing
# ----------------------------
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)
english_stopwords = frozenset(stopwords.words("english"))
corpus_stopwords = {
    "category","references","also","external","links","may","first","see","history",
    "people","one","two","part","thumb","including","second","following","many","however",
    "would","became"
}
ALL_STOPWORDS = english_stopwords.union(corpus_stopwords)

RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

def tokenize(text: str):
    return [m.group() for m in RE_WORD.finditer(text.lower()) if m.group() not in ALL_STOPWORDS]

# ----------------------------
# Load data (indices + PageRank + PageViews + titles)
# ----------------------------
print("Loading indices and auxiliary data...")

AUX_DIR = os.environ.get("AUX_DIR", "").strip("/")
# --- Memory-aware loading flags (keep the VM alive) ---
def _mem_total_gb():
    """Best-effort total RAM in GB (Linux). Returns None if unknown."""
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb / 1024 / 1024
    except Exception:
        return None
    return None

_MEM_GB = _mem_total_gb()
_LOW_MEM_GB = float(os.environ.get("LOW_MEM_GB", "8"))  # treat <8GB as "low memory" by default
_LOW_MEM = (_MEM_GB is not None) and (_MEM_GB < _LOW_MEM_GB)

def _env_flag(name: str, default: str) -> bool:
    """Parse env var booleans with 'auto' support."""
    v = os.environ.get(name, default).strip().lower()
    if v in ("1", "true", "yes", "y", "on"):
        return True
    if v in ("0", "false", "no", "n", "off"):
        return False
    if v == "auto":
        return not _LOW_MEM
    return default.strip().lower() in ("1","true","yes","y","on")

# Heavy aux structures:
# - Titles mapping (doc_id -> title) is massive.
# - PageRank / PageViews can also be massive.
# On small VMs, loading them eagerly tends to trigger OOM.
LOAD_TITLES   = _env_flag("LOAD_TITLES",   "auto")
LOAD_PAGERANK = _env_flag("LOAD_PAGERANK", "auto")
LOAD_PAGEVIEWS= _env_flag("LOAD_PAGEVIEWS","auto")

# If you want to keep startup light but still allow later loading (at your own risk),
# set LAZY_AUX=1 (default). Setting LAZY_AUX=0 will disable later loading too.
LAZY_AUX = _env_flag("LAZY_AUX", "1")

# A short, safe list of likely locations for auxiliary pickles in your bucket.
# We intentionally do NOT list the whole bucket (could be huge due to posting shards).
_SEARCH_PREFIXES = []
if AUX_DIR:
    _SEARCH_PREFIXES.append(AUX_DIR + "/")
_SEARCH_PREFIXES += [
    "",  # exact path as provided
    "aux/", "auxiliary/", "data/", "resources/", "static/", "pickles/", "pkl/", "output/",
    f"{BODY_DIR}/", f"{TITLE_DIR}/", f"{ANCHOR_DIR}/",
]

_AUX_SOURCES = {}  # filename -> where it was loaded from (local path or GCS blob name)

def _load_pickle_smart(local_name: str, gcs_candidates=None, default=None):
    """Load a pickle either locally or from GCS.

    Why this exists:
    - On the VM you might not have the aux pkls locally.
    - In the bucket, those pkls might live under a folder (e.g., aux/pageviews.pkl).
    - We avoid listing the whole bucket (posting files can be enormous).
    """
    if gcs_candidates is None:
        gcs_candidates = [local_name]

    # 1) Local (exact path)
    try:
        with open(local_name, "rb") as f:
            obj = pickle.load(f)
            _AUX_SOURCES[local_name] = f"local:{local_name}"
            return obj
    except Exception:
        pass

    # 2) Local (basename in cwd)
    base = os.path.basename(local_name)
    if base != local_name:
        try:
            with open(base, "rb") as f:
                obj = pickle.load(f)
                _AUX_SOURCES[local_name] = f"local:{base}"
                return obj
        except Exception:
            pass

    # 3) GCS (explicit candidates)
    for cand in gcs_candidates:
        if not cand:
            continue
        try:
            bucket = _get_bucket()
            blob = bucket.blob(cand)
            if blob.exists():
                with blob.open("rb") as f:
                    obj = pickle.load(f)
                _AUX_SOURCES[local_name] = f"gcs:{cand}"
                return obj
        except Exception:
            # Keep trying other candidates/prefixes
            pass

    # 4) GCS (try common prefixes + basename)
    for pref in _SEARCH_PREFIXES:
        if not pref:
            continue
        cand = pref + base
        try:
            bucket = _get_bucket()
            blob = bucket.blob(cand)
            if blob.exists():
                with blob.open("rb") as f:
                    obj = pickle.load(f)
                _AUX_SOURCES[local_name] = f"gcs:{cand}"
                return obj
        except Exception:
            pass

    _AUX_SOURCES[local_name] = "missing"
    return default

# Indices
try:
    idx_body = InvertedIndex.read_index(BODY_DIR, BODY_INDEX_NAME)
except Exception:
    # fallback to "index.pkl" if you used the staff naming
    idx_body = InvertedIndex.read_index(BODY_DIR, "index")

try:
    idx_title = InvertedIndex.read_index(TITLE_DIR, TITLE_INDEX_NAME)
except Exception:
    idx_title = InvertedIndex.read_index(TITLE_DIR, "index")

try:
    idx_anchor = InvertedIndex.read_index(ANCHOR_DIR, ANCHOR_INDEX_NAME)
except Exception:
    # Anchor is optional in some builds; keep empty if missing.
    idx_anchor = None

# Dictionaries
# Dictionaries (optionally loaded to save RAM)
page_rank = {}
page_views = {}
id_to_title = {}

def _ensure_pagerank_loaded():
    global page_rank, MAX_PR
    if page_rank:
        return
    if not LAZY_AUX or not LOAD_PAGERANK:
        return
    pr = _load_pickle_smart("pr.pkl", [os.environ.get("PAGERANK_BLOB", "pr.pkl")], default={})
    if isinstance(pr, dict):
        page_rank = pr
        MAX_PR = max(page_rank.values()) if len(page_rank) else 0.0

def _ensure_pageviews_loaded():
    global page_views, MAX_PV
    if page_views:
        return
    if not LAZY_AUX or not LOAD_PAGEVIEWS:
        return
    pv = _load_pickle_smart("pageviews.pkl", [os.environ.get("PAGEVIEWS_BLOB", "pageviews.pkl")], default={})
    if isinstance(pv, dict):
        page_views = pv
        MAX_PV = max(page_views.values()) if len(page_views) else 0

def _ensure_titles_loaded():
    global id_to_title, N_DOCS
    if id_to_title:
        return
    if not LAZY_AUX or not LOAD_TITLES:
        return
    tt = _load_pickle_smart("id_to_title.pkl", [os.environ.get("TITLES_BLOB", "id_to_title.pkl")], default={})
    if isinstance(tt, dict):
        id_to_title = tt
        if len(id_to_title) > 0:
            N_DOCS = len(id_to_title)

# Eager-load only what we are allowed to on this machine
if LOAD_PAGERANK:
    _ensure_pagerank_loaded()
if LOAD_PAGEVIEWS:
    _ensure_pageviews_loaded()
if LOAD_TITLES:
    _ensure_titles_loaded()

# Corpus size for IDF (best-effort)
# If you have titles mapping for all docs, this is accurate; otherwise we fall back to a known constant.
# Corpus size for IDF (best-effort).
# Keep this constant unless titles are loaded and cover the corpus.
N_DOCS = 6_348_910

# Max values for normalization (computed when the dicts are loaded)
MAX_PR = max(page_rank.values()) if isinstance(page_rank, dict) and len(page_rank) else 0.0
MAX_PV = max(page_views.values()) if isinstance(page_views, dict) and len(page_views) else 0

print("Loaded:",
      f"body_terms={len(getattr(idx_body,'df',{})):,}",
      f"title_terms={len(getattr(idx_title,'df',{})):,}",
      f"anchor={'ok' if idx_anchor is not None else 'missing'},",
      f"pagerank={len(page_rank):,}, pageviews={len(page_views):,}, titles={len(id_to_title):,}")
print("Load flags:",
      f"RAM≈{_MEM_GB:.1f}GB" if _MEM_GB is not None else "RAM=?",
      f"LOW_MEM={_LOW_MEM}",
      f"LOAD_TITLES={LOAD_TITLES}",
      f"LOAD_PAGERANK={LOAD_PAGERANK}",
      f"LOAD_PAGEVIEWS={LOAD_PAGEVIEWS}",
      f"LAZY_AUX={LAZY_AUX}")
print("Aux sources:",
      _AUX_SOURCES.get("pr.pkl", "?"),
      _AUX_SOURCES.get("pageviews.pkl", "?"),
      _AUX_SOURCES.get("id_to_title.pkl", "?"))

# ----------------------------
# Ranking methods
# ----------------------------
def _idf(df: int, N: int) -> float:
    # Standard smooth IDF; avoids div-by-zero.
    return math.log10((N + 1) / (df + 1))

def search_body_cosine(query_tokens):
    """Cosine similarity using TF-IDF on body (implemented over query-term subspace).

    Returns: Counter(doc_id -> score)
    """
    if not query_tokens:
        return Counter()

    q_tf = Counter(query_tokens)
    q_weights = {}
    for t, tf in q_tf.items():
        if t in idx_body.df:
            q_weights[t] = tf * _idf(idx_body.df[t], N_DOCS)

    if not q_weights:
        return Counter()

    q_norm = math.sqrt(sum(w * w for w in q_weights.values())) or 1.0

    dot = Counter()
    doc_norm_sq = Counter()

    for t, w_tq in q_weights.items():
        pls = read_posting_list(idx_body, BODY_DIR, t)
        idf_t = _idf(idx_body.df[t], N_DOCS)
        for doc_id, tf in pls:
            w_td = tf * idf_t
            dot[doc_id] += w_td * w_tq
            doc_norm_sq[doc_id] += w_td * w_td

    scores = Counter()
    for doc_id, numerator in dot.items():
        denom = (math.sqrt(doc_norm_sq[doc_id]) or 1.0) * q_norm
        scores[doc_id] = numerator / denom

    return scores


def search_title_binary(query_tokens):
    """Binary ranking on titles: score = number of query terms that appear in title."""
    if not query_tokens:
        return Counter()
    scores = Counter()
    uniq = set(query_tokens)
    for t in uniq:
        pls = read_posting_list(idx_title, TITLE_DIR, t)
        for doc_id, _tf in pls:
            scores[doc_id] += 1
    return scores


def search_anchor_binary(query_tokens):
    """Binary ranking on anchor text: score = number of query terms that appear in anchors."""
    if idx_anchor is None or not query_tokens:
        return Counter()
    scores = Counter()
    uniq = set(query_tokens)
    for t in uniq:
        if t not in idx_anchor.df:
            continue
        pls = read_posting_list(idx_anchor, ANCHOR_DIR, t)
        for doc_id, _tf in pls:
            scores[doc_id] += 1
    return scores


def _norm01(x: float, xmax: float) -> float:
    return (x / xmax) if (xmax and x > 0) else 0.0

def _norm_log(x: float, xmax: float) -> float:
    if not xmax or x <= 0:
        return 0.0
    return math.log1p(x) / math.log1p(xmax)

# ----------------------------
# Flask app
# ----------------------------
class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super().run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
# NOTE (Colab compatibility): the staff notebook does `import search_frontend as se` and then calls
# `se.run(...)`. In that case, `se` is the *module*, so it must expose a module-level `run` function.
# We therefore provide `run()` that proxies to the Flask app.
def run(*args, **kwargs):
    """Run the Flask app (module-level helper for Colab notebooks)."""
    return app.run(*args, **kwargs)

# Optional: expose the Flask app object as well (useful if someone wants `search_frontend.app`).
se = app
app.config["JSONIFY_PRETTYPRINT_REGULAR"] = False


@app.route("/search")
def search():
    """Main search: returns top 100 results using combined ranking.

    Combines (when enabled/available):
      - Cosine similarity (body TF-IDF)
      - Binary title match
      - Binary anchor match
      - PageRank
      - PageViews
    """
    query = request.args.get("query", "")
    if not query:
        return jsonify([])

    tokens = tokenize(query)

    body_scores = search_body_cosine(tokens)
    title_scores = search_title_binary(tokens)
    anchor_scores = search_anchor_binary(tokens)

    candidates = set(body_scores.keys()) | set(title_scores.keys()) | set(anchor_scores.keys())
    if not candidates:
        return jsonify([])

    # Lazy-load aux only if the current machine allows it.
    if LOAD_PAGERANK:
        _ensure_pagerank_loaded()
    if LOAD_PAGEVIEWS:
        _ensure_pageviews_loaded()
    if LOAD_TITLES:
        _ensure_titles_loaded()

    q_len = max(1, len(set(tokens)))

    # Base weights (will be renormalized if some features are unavailable)
    weights = {
        "body": 0.60,
        "title": 0.20,
        "anchor": 0.10,
        "pr": 0.05 if (MAX_PR and page_rank) else 0.0,
        "pv": 0.05 if (MAX_PV and page_views) else 0.0,
    }
    w_sum = sum(weights.values()) or 1.0
    for k in weights:
        weights[k] = weights[k] / w_sum

    final_scores = Counter()
    for doc_id in candidates:
        s_body = body_scores.get(doc_id, 0.0)                    # ~ [0,1]
        s_title = title_scores.get(doc_id, 0) / q_len            # [0,1]
        s_anchor = anchor_scores.get(doc_id, 0) / q_len          # [0,1]
        s_pr = _norm01(page_rank.get(doc_id, 0.0), MAX_PR)       # [0,1]
        s_pv = _norm_log(page_views.get(doc_id, 0), MAX_PV)      # [0,1]

        score = (weights["body"] * s_body) + (weights["title"] * s_title) + (weights["anchor"] * s_anchor)                 + (weights["pr"] * s_pr) + (weights["pv"] * s_pv)
        final_scores[doc_id] = score

    top_100 = final_scores.most_common(100)

    # Titles: if not loaded (or missing a doc), fall back to doc_id string.
    res = [(int(doc_id), id_to_title.get(doc_id, str(doc_id))) for doc_id, _ in top_100]
    return jsonify(res)


@app.route("/search_body")

def search_body():
    """Cosine Similarity + TF-IDF on article body."""
    query = request.args.get("query", "")
    if not query:
        return jsonify([])

    tokens = tokenize(query)
    scores = search_body_cosine(tokens)
    top_100 = scores.most_common(100)
    if LOAD_TITLES:
        _ensure_titles_loaded()
    res = [(int(doc_id), id_to_title.get(doc_id, str(doc_id))) for doc_id, _ in top_100]
    return jsonify(res)


@app.route("/search_title")
def search_title():
    """Binary search over titles."""
    query = request.args.get("query", "")
    if not query:
        return jsonify([])

    tokens = tokenize(query)
    scores = search_title_binary(tokens)
    top_100 = scores.most_common(100)
    if LOAD_TITLES:
        _ensure_titles_loaded()
    res = [(int(doc_id), id_to_title.get(doc_id, str(doc_id))) for doc_id, _ in top_100]
    return jsonify(res)


@app.route("/search_anchor")
def search_anchor():
    """Binary search over anchor text."""
    query = request.args.get("query", "")
    if not query:
        return jsonify([])

    tokens = tokenize(query)
    scores = search_anchor_binary(tokens)
    top_100 = scores.most_common(100)
    if LOAD_TITLES:
        _ensure_titles_loaded()
    res = [(int(doc_id), id_to_title.get(doc_id, str(doc_id))) for doc_id, _ in top_100]
    return jsonify(res)


@app.route("/get_pagerank", methods=["POST"])
def get_pagerank():
    """Return PageRank scores for a list of wiki_ids."""
    wiki_ids = request.get_json() or []
    if LOAD_PAGERANK:
        _ensure_pagerank_loaded()
    res = [float(page_rank.get(int(doc_id), 0.0)) for doc_id in wiki_ids]
    return jsonify(res)


@app.route("/get_pageview", methods=["POST"])
def get_pageview():
    """Return PageViews for a list of wiki_ids."""
    wiki_ids = request.get_json() or []
    if LOAD_PAGEVIEWS:
        _ensure_pageviews_loaded()
    res = [int(page_views.get(int(doc_id), 0)) for doc_id in wiki_ids]
    return jsonify(res)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")), debug=False, use_reloader=False)
