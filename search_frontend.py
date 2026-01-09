from flask import Flask, request, jsonify
import sys
from collections import Counter, OrderedDict, defaultdict
import itertools
from pathlib import Path
import pickle
from google.cloud import storage
import math
import re
import nltk
from nltk.corpus import stopwords


class MultiFileReader:
    """ Sequential binary reader of multiple files of up to BLOCK_SIZE each. """

    def __init__(self, base_dir, bucket_name=None):
        self._base_dir = Path(base_dir)
        self._open_files = {}

    def read(self, locs, n_bytes):
        b = []
        for f_name, offset in locs:
            f_name = str(self._base_dir / f_name)
            if f_name not in self._open_files:
                self._open_files[f_name] = open(f_name, 'rb')
            f = self._open_files[f_name]
            f.seek(offset)
            n_read = min(n_bytes, 1999998 - offset)
            b.append(f.read(n_read))
            n_bytes -= n_read
        return b''.join(b)

    def close(self):
        for f in self._open_files.values():
            f.close()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


TUPLE_SIZE = 6
TF_MASK = 2 ** 16 - 1


class InvertedIndex:
    def __init__(self, docs={}):
        self.df = Counter()
        self.term_total = Counter()
        self.posting_locs = defaultdict(list)
        for doc_id, tokens in docs.items():
            self.add_doc(doc_id, tokens)

    @staticmethod
    def read_index(base_dir, name):
        with open(Path(base_dir) / f'{name}.pkl', 'rb') as f:
            return pickle.load(f)


def read_posting_list(index, base_dir, w):
    with MultiFileReader(base_dir) as reader:
        if w not in index.posting_locs:
            return []
        locs = index.posting_locs[w]
        b = reader.read(locs, index.df[w] * TUPLE_SIZE)
        posting_list = []
        for i in range(index.df[w]):
            doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
            tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
            posting_list.append((doc_id, tf))
        return posting_list


# --- BM25 Class (CORRECTED) ---
class BM25_from_index:
    def __init__(self, index, base_dir="postings_gcp_body", k1=1.5, b=0.75):
        self.index = index
        self.base_dir = base_dir
        self.k1 = k1
        self.b = b
        # FIX: N must be the total number of documents in the corpus, not the number of terms!
        self.N = 6348910
        self.avgdl = 300

    def calc_idf(self, list_of_tokens):
        idf = {}
        for term in list_of_tokens:
            if term in self.index.df:
                n_ti = self.index.df[term]
                idf[term] = math.log(1 + (self.N - n_ti + 0.5) / (n_ti + 0.5))
            else:
                pass
        return idf

    def search(self, query):
        idf = self.calc_idf(query)
        scores = Counter()
        for term in query:
            if term not in self.index.df:
                continue
            pls = read_posting_list(self.index, self.base_dir, term)
            current_idf = idf.get(term, 0)
            for doc_id, tf in pls:
                numerator = current_idf * tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (1))
                score = numerator / denominator
                scores[doc_id] += score
        return scores


# --- Data Loading & Preprocessing ---

nltk.download('stopwords')
english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

print("Loading indices and data...")
# Load indices
idx_body = InvertedIndex.read_index("postings_gcp_body", "index")
idx_title = InvertedIndex.read_index("postings_gcp_title", "index")

# Load PageRank and PageViews
try:
    with open("pr.pkl", "rb") as f:
        page_rank = pickle.load(f)
except FileNotFoundError:
    print("Warning: pr.pkl not found. PageRank scores will be 0.")
    page_rank = {}

try:
    with open("pageviews.pkl", "rb") as f:
        page_views = pickle.load(f)
except FileNotFoundError:
    print("Warning: pageviews.pkl not found. PageView scores will be 0.")
    page_views = {}

# Load ID to Title mapping (CRITICAL FIX)
try:
    with open("id_to_title.pkl", "rb") as f:
        id_to_title = pickle.load(f)
except FileNotFoundError:
    print("Warning: id_to_title.pkl not found. Results will show IDs instead of titles.")
    id_to_title = {}

bm25_body = BM25_from_index(idx_body, base_dir="postings_gcp_body")
print("Data loaded!")


def tokenize(text):
    return [token.group() for token in RE_WORD.finditer(text.lower()) if token.group() not in all_stopwords]


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


@app.route("/search")
def search():
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)

    tokens = tokenize(query)

    # 1. Body Search (BM25)
    body_scores = bm25_body.search(tokens)

    # 2. Title Search (Binary / Count)
    title_scores = Counter()
    for term in tokens:
        pls = read_posting_list(idx_title, "postings_gcp_title", term)
        for doc_id, tf in pls:
            title_scores[doc_id] += 1

    final_scores = Counter()
    candidates = set(body_scores.keys()) | set(title_scores.keys())

    for doc_id in candidates:
        s_body = body_scores.get(doc_id, 0)
        s_title = title_scores.get(doc_id, 0)
        s_pr = page_rank.get(doc_id, 0)
        s_pv = page_views.get(doc_id, 0)

        # Log scaling for page views to avoid dominance
        log_pv = math.log(s_pv + 1, 10) if s_pv > 0 else 0

        # Adjusted weights for Minimum Requirements:
        # BM25 is usually 10-50, Title is small (1-3), PR is tiny (0.0001).
        # We boost Title and PR significantly.
        score = (1.0 * s_body) + (5.0 * s_title) + (1.0 * log_pv) + (1000.0 * s_pr)
        final_scores[doc_id] = score

    top_100 = final_scores.most_common(100)

    # Return (ID, Title) instead of (ID, ID)
    res = [(str(doc_id), id_to_title.get(doc_id, str(doc_id))) for doc_id, score in top_100]

    return jsonify(res)


@app.route("/search_body")
def search_body():
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    tokens = tokenize(query)
    scores = bm25_body.search(tokens)
    top_100 = scores.most_common(100)
    res = [(str(doc_id), id_to_title.get(doc_id, str(doc_id))) for doc_id, score in top_100]
    return jsonify(res)


@app.route("/search_title")
def search_title():
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    tokens = tokenize(query)
    scores = Counter()
    for term in tokens:
        pls = read_posting_list(idx_title, "postings_gcp_title", term)
        for doc_id, tf in pls:
            scores[doc_id] += 1

    sorted_res = scores.most_common()
    res = [(str(doc_id), id_to_title.get(doc_id, str(doc_id))) for doc_id, score in sorted_res]
    return jsonify(res)


@app.route("/search_anchor")
def search_anchor():
    # Leaving empty as per Minimum Requirements strategy
    # (assuming you didn't build the anchor index to save time/resources)
    res = []
    return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    res = [page_rank.get(int(doc_id), 0) for doc_id in wiki_ids]
    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    res = [page_views.get(int(doc_id), 0) for doc_id in wiki_ids]
    return jsonify(res)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)