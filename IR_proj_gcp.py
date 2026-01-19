#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
from collections import Counter, OrderedDict, defaultdict
import itertools
from itertools import islice, count, groupby
import pandas as pd
import os
import re
from operator import itemgetter
from time import time
from pathlib import Path
import pickle
from google.cloud import storage
from contextlib import closing
import hashlib
from pyspark.sql import SparkSession

PROJECT_ID = 'irproject2026'
BUCKET_NAME = 'irprojectbucket'

# INVERTED INDEX
def get_bucket(bucket_name):
    return storage.Client(PROJECT_ID).bucket(bucket_name)

def _open(path, mode, bucket=None):
    if bucket is None:
        return open(path, mode)
    return bucket.blob(path).open(mode)

BLOCK_SIZE = 1999998

class MultiFileWriter:
    def __init__(self, base_dir, name, bucket_name=None):
        self._base_dir = Path(base_dir)
        self._name = name
        self._bucket = None if bucket_name is None else get_bucket(bucket_name)
        self._file_gen = (_open(str(self._base_dir / f'{name}_{i:03}.bin'),
                                'wb', self._bucket)
                          for i in itertools.count())
        self._f = next(self._file_gen)

    def write(self, b):
        locs = []
        while len(b) > 0:
            pos = self._f.tell()
            remaining = BLOCK_SIZE - pos
            if remaining == 0:
                self._f.close()
                self._f = next(self._file_gen)
                pos, remaining = 0, BLOCK_SIZE
            self._f.write(b[:remaining])
            name = self._f.name if hasattr(self._f, 'name') else self._f._blob.name
            locs.append((name, pos))
            b = b[remaining:]
        return locs

    def close(self):
        self._f.close()

class MultiFileReader:
    def __init__(self, base_dir, bucket_name=None):
        self._base_dir = Path(base_dir)
        self._bucket = None if bucket_name is None else get_bucket(bucket_name)
        self._open_files = {}

    def read(self, locs, n_bytes):
        b = []
        for f_name, offset in locs:
            f_name = str(self._base_dir / f_name)
            if f_name not in self._open_files:
                self._open_files[f_name] = _open(f_name, 'rb', self._bucket)
            f = self._open_files[f_name]
            f.seek(offset)
            n_read = min(n_bytes, BLOCK_SIZE - offset)
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
        self._posting_list = defaultdict(list)
        self.posting_locs = defaultdict(list)
        for doc_id, tokens in docs.items():
            self.add_doc(doc_id, tokens)

    def add_doc(self, doc_id, tokens):
        w2cnt = Counter(tokens)
        self.term_total.update(w2cnt)
        for w, cnt in w2cnt.items():
            self.df[w] = self.df.get(w, 0) + 1
            self._posting_list[w].append((doc_id, cnt))

    def write_index(self, base_dir, name, bucket_name=None):
        if not os.path.exists(base_dir):
            os.makedirs(base_dir, exist_ok=True)
        
        # Write global index locally
        path = str(Path(base_dir) / f'{name}.pkl')
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        
        # Upload if bucket provided
        if bucket_name:
            client = storage.Client(PROJECT_ID)
            bucket = client.bucket(bucket_name)
            blob_path = f"{base_dir}/{name}.pkl"
            bucket.blob(blob_path).upload_from_filename(path)
            # Cleanup
            try:
                os.remove(path)
            except:
                pass

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_posting_list']
        return state

    @staticmethod
    def write_a_posting_list(b_w_pl, base_dir, bucket_name=None):
        posting_locs = defaultdict(list)
        bucket_id, list_w_pl = b_w_pl

        # Ensure local dir exists
        if not os.path.exists(base_dir):
            os.makedirs(base_dir, exist_ok=True)

        with closing(MultiFileWriter(base_dir, bucket_id, bucket_name)) as writer:
            for w, pl in list_w_pl:
                b = b''.join([(doc_id << 16 | (tf & TF_MASK)).to_bytes(TUPLE_SIZE, 'big')
                              for doc_id, tf in pl])
                locs = writer.write(b)
                posting_locs[w].extend(locs)
            
            # Save posting_locs dict locally first
            path = str(Path(base_dir) / f'{bucket_id}_posting_locs.pickle')
            with open(path, 'wb') as f:
                pickle.dump(posting_locs, f)
            
            # Upload if bucket
            if bucket_name:
                bucket = get_bucket(bucket_name)
                blob_path = f"{base_dir}/{bucket_id}_posting_locs.pickle"
                bucket.blob(blob_path).upload_from_filename(path)
                try:
                    os.remove(path)
                except:
                    pass
                    
        return bucket_id

# HELPERS

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

NUM_BUCKETS = 124


def _hash(s):
    return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()

def token2bucket_id(token):
  return int(_hash(token),16) % NUM_BUCKETS

def word_count(text, id):
  tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
  filtered_tokens = [token for token in tokens if token not in all_stopwords]
  word_counts = Counter(filtered_tokens)
  result = [(word, (id, count)) for word, count in word_counts.items()]
  return result

def reduce_word_counts(unsorted_pl):
  return sorted(unsorted_pl, key=lambda x: x[0])

def calculate_df(postings):
  return postings.mapValues(len)


def partition_postings_and_write(postings, bucket_name, folder_name):
  postings = postings.filter(lambda x: len(x[1]) < 2000000)
  bucketed_postings = postings.map(lambda x: (token2bucket_id(x[0]), (x[0], x[1])))
  grouped_by_bucket = bucketed_postings.groupByKey()
  posting_locs = grouped_by_bucket.map(
      lambda x: InvertedIndex.write_a_posting_list((x[0], list(x[1])), folder_name, bucket_name)
  )
  return posting_locs


# In[ ]:


import sys
from collections import Counter, OrderedDict, defaultdict
import itertools
from itertools import islice, count, groupby
import pandas as pd
import os
import re
from operator import itemgetter
from time import time
from pathlib import Path
import pickle
from google.cloud import storage
from contextlib import closing
import hashlib
from pyspark.sql import SparkSession

# --- CONFIGURATION ---
PROJECT_ID = 'irproject2026'
BUCKET_NAME = 'irprojectbucket'

# Paths
WIKI_PATH = f"gs://{BUCKET_NAME}/*.parquet" 
PV_PATH = f"gs://{BUCKET_NAME}/pageviews-202108-user.bz2"

# --- INIT SPARK ---
spark = SparkSession.builder     .appName("IR_Project_Index")     .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.1-s_2.12")     .config("spark.driver.memory", "8g")     .config("spark.executor.memory", "8g")     .config("spark.master", "local[*]")     .getOrCreate()

# Load Data
parquetFile = spark.read.parquet(WIKI_PATH)

# --- UTILS & CLASSES ---

def get_bucket(bucket_name):
    return storage.Client(PROJECT_ID).bucket(bucket_name)

BLOCK_SIZE = 1999998

class MultiFileWriter:
    def __init__(self, base_dir, name, bucket_name=None):
        self._base_dir = Path(base_dir)
        self._name = name
        self._bucket = None if bucket_name is None else get_bucket(bucket_name)
        self._file_gen = (self._open_local_file(i) for i in itertools.count())
        self._f, self._current_filename = next(self._file_gen)

    def _open_local_file(self, i):
        filename = f'{self._name}_{i:03}.bin'
        path = str(self._base_dir / filename)
        if not os.path.exists(self._base_dir):
            os.makedirs(self._base_dir, exist_ok=True)
        return open(path, 'wb'), filename

    def _upload_and_close_current(self):
        if self._f:
            filepath = self._f.name
            self._f.close()
            if self._bucket:
                blob_path = f"{self._base_dir}/{self._current_filename}"
                blob = self._bucket.blob(blob_path)
                blob.upload_from_filename(filepath)
                try:
                    os.remove(filepath)
                except:
                    pass

    def write(self, b):
        locs = []
        while len(b) > 0:
            pos = self._f.tell()
            remaining = BLOCK_SIZE - pos
            if remaining == 0:
                self._upload_and_close_current()
                self._f, self._current_filename = next(self._file_gen)
                pos, remaining = 0, BLOCK_SIZE
            self._f.write(b[:remaining])
            locs.append((self._current_filename, pos))
            b = b[remaining:]
        return locs

    def close(self):
        self._upload_and_close_current()

TUPLE_SIZE = 6
TF_MASK = 2 ** 16 - 1

class InvertedIndex:
    def __init__(self, docs={}):
        self.df = Counter()
        self.term_total = Counter()
        self._posting_list = defaultdict(list)
        self.posting_locs = defaultdict(list)
        for doc_id, tokens in docs.items():
            self.add_doc(doc_id, tokens)

    def add_doc(self, doc_id, tokens):
        w2cnt = Counter(tokens)
        self.term_total.update(w2cnt)
        for w, cnt in w2cnt.items():
            self.df[w] = self.df.get(w, 0) + 1
            self._posting_list[w].append((doc_id, cnt))

    def write_index(self, base_dir, name, bucket_name=None):
        if not os.path.exists(base_dir):
            os.makedirs(base_dir, exist_ok=True)
            
        # Write global index locally
        path = str(Path(base_dir) / f'{name}.pkl')
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        
        # Upload if bucket provided
        if bucket_name:
            client = storage.Client(PROJECT_ID)
            bucket = client.bucket(bucket_name)
            blob_path = f"{base_dir}/{name}.pkl"
            bucket.blob(blob_path).upload_from_filename(path)
            try:
                os.remove(path)
            except:
                pass

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_posting_list']
        return state

    @staticmethod
    def write_a_posting_list(b_w_pl, base_dir, bucket_name=None):
        posting_locs = defaultdict(list)
        bucket_id, list_w_pl = b_w_pl

        if not os.path.exists(base_dir):
            os.makedirs(base_dir, exist_ok=True)

        with closing(MultiFileWriter(base_dir, bucket_id, bucket_name)) as writer:
            for w, pl in list_w_pl:
                b = b''.join([(doc_id << 16 | (tf & TF_MASK)).to_bytes(TUPLE_SIZE, 'big')
                              for doc_id, tf in pl])
                locs = writer.write(b)
                posting_locs[w].extend(locs)
            
            path = str(Path(base_dir) / f'{bucket_id}_posting_locs.pickle')
            with open(path, 'wb') as f:
                pickle.dump(posting_locs, f)
            
            if bucket_name:
                bucket = get_bucket(bucket_name)
                blob_path = f"{base_dir}/{bucket_id}_posting_locs.pickle"
                bucket.blob(blob_path).upload_from_filename(path)
                try:
                    os.remove(path)
                except:
                    pass
                    
        return bucket_id

# --- NLP HELPERS ---
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

NUM_BUCKETS = 124

def _hash(s):
    return hashlib.blake2b(bytes(s, encoding='utf8'), digest_size=5).hexdigest()

def token2bucket_id(token):
  return int(_hash(token),16) % NUM_BUCKETS

def word_count(text, id):
  tokens = [token.group() for token in RE_WORD.finditer(text.lower())]
  filtered_tokens = [token for token in tokens if token not in all_stopwords]
  word_counts = Counter(filtered_tokens)
  result = [(word, (id, count)) for word, count in word_counts.items()]
  return result

def reduce_word_counts(unsorted_pl):
  return sorted(unsorted_pl, key=lambda x: x[0])

def calculate_df(postings):
  return postings.mapValues(len)

def partition_postings_and_write(postings, bucket_name, folder_name):
  postings = postings.filter(lambda x: len(x[1]) < 2000000)
  bucketed_postings = postings.map(lambda x: (token2bucket_id(x[0]), (x[0], x[1])))
  grouped_by_bucket = bucketed_postings.groupByKey()
  posting_locs = grouped_by_bucket.map(
      lambda x: InvertedIndex.write_a_posting_list((x[0], list(x[1])), folder_name, bucket_name)
  )
  return posting_locs

# --- CREATE INDEX FUNCTION (FIXED) ---
def create_index(source_col, index_folder_name, is_anchor=False):
    print(f"Starting index creation for: {source_col if source_col else 'Anchor'} -> {index_folder_name}")
    
    if is_anchor:
        rdd_pairs = parquetFile.select("id", "anchor_text").rdd
        anchor_pairs = rdd_pairs.flatMap(lambda row: [(link.id, link.text) for link in row.anchor_text])
        grouped_anchors = anchor_pairs.groupByKey().mapValues(lambda texts: " ".join(texts))
        word_counts = grouped_anchors.flatMap(lambda x: word_count(x[1], x[0]))
    else:
        rdd_pairs = parquetFile.select(source_col, "id").rdd
        word_counts = rdd_pairs.flatMap(lambda x: word_count(x[0], x[1]))
    
    postings = word_counts.groupByKey().mapValues(reduce_word_counts)
    
    # Filter rare words to save memory
    if source_col == "text":
        postings = postings.filter(lambda x: len(x[1]) > 50)
    
    w2df = calculate_df(postings)
    w2df_dict = w2df.collectAsMap()
    
    _ = partition_postings_and_write(postings, BUCKET_NAME, index_folder_name).collect()
    
    super_posting_locs = defaultdict(list)
    client = storage.Client(PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)
    
    for blob in bucket.list_blobs(prefix=index_folder_name):
        if not blob.name.endswith("pickle"):
            continue
        
        local_temp = f"temp_{blob.name.split('/')[-1]}"
        blob.download_to_filename(local_temp)
        
        with open(local_temp, "rb") as f:
            posting_locs = pickle.load(f)
            for k, v in posting_locs.items():
                super_posting_locs[k].extend(v)
        try:
            os.remove(local_temp)
        except:
            pass

    inverted = InvertedIndex()
    inverted.posting_locs = super_posting_locs
    inverted.df = w2df_dict

    if not os.path.exists(index_folder_name):
        os.makedirs(index_folder_name, exist_ok=True)

    local_index_name = f'index_{index_folder_name.split("_")[-1]}.pkl'
    inverted.write_index(index_folder_name, local_index_name.replace('.pkl', ''), BUCKET_NAME)
    
    print(f"Finished creating index for {index_folder_name}.")

# --- EXECUTION ---

# 1. Body Index
create_index("text", "postings_gcp_body")

# 2. Title Index
create_index("title", "postings_gcp_title")

# 3. Anchor Text Index
create_index(None, "postings_gcp_anchor", is_anchor=True)

# --- PAGERANK ---
print("Starting PageRank...")
from graphframes import *

def generate_graph(pages):
    edges = pages.flatMap(lambda row: [(row.id, link.id) for link in row.anchor_text]).distinct()
    vertices = pages.flatMap(lambda row: [(row.id,) for link in row.anchor_text] + [(row.id,)]).distinct().map(lambda x: (x[0], x[0]))
    return edges, vertices

pages_links = parquetFile.select("id", "anchor_text").rdd
edges_rdd, vertices_rdd = generate_graph(pages_links)

edgesDF = edges_rdd.toDF(["src", "dst"]).repartition(124, "src")
verticesDF = vertices_rdd.toDF(["id", "title"]).repartition(124, "id")

g = GraphFrame(verticesDF, edgesDF)
pr_results = g.pageRank(resetProbability=0.15, maxIter=10)
pr_rdd = pr_results.vertices.select("id", "pagerank").rdd
pr_dict = pr_rdd.collectAsMap()

with open('pr.pkl', 'wb') as f:
    pickle.dump(pr_dict, f)

os.system(f"gsutil cp pr.pkl gs://{BUCKET_NAME}/pr.pkl")
print("PageRank finished.")

# --- PAGEVIEWS ---
print("Starting PageViews...")
pv_df = spark.read.text(PV_PATH)

def parse_page_views(line):
    parts = line.value.split(" ")
    try:
        page_id = int(parts[0])
        views = int(parts[2])
        return (page_id, views)
    except:
        return None

pv_rdd = pv_df.rdd.map(parse_page_views).filter(lambda x: x is not None)
pv_dict = pv_rdd.collectAsMap()

with open('pageviews.pkl', 'wb') as f:
    pickle.dump(pv_dict, f)

os.system(f"gsutil cp pageviews.pkl gs://{BUCKET_NAME}/pageviews.pkl")
print("PageViews finished.")

# --- TITLE MAPPING ---
print("Starting ID to Title mapping...")
def create_id_to_title_mapping():
    df = parquetFile.select("id", "title")
    id_to_title = df.rdd.collectAsMap()

    with open('id_to_title.pkl', 'wb') as f:
        pickle.dump(id_to_title, f)

    os.system(f"gsutil cp id_to_title.pkl gs://{BUCKET_NAME}/id_to_title.pkl")

create_id_to_title_mapping()
print("All tasks finished successfully!")


# In[ ]:


# --- PAGERANK ---
import sys
import os
import pickle
from pyspark.sql import SparkSession
from google.cloud import storage

get_ipython().system('pip install graphframes')
from graphframes import *

PROJECT_ID = 'irproject2026'
BUCKET_NAME = 'irprojectbucket'
WIKI_PATH = f"gs://{BUCKET_NAME}/*.parquet"
PV_PATH = f"gs://{BUCKET_NAME}/pageviews-202108-user.bz2"

spark = SparkSession.builder     .appName("PageRank_Light")     .config("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.1-s_2.12")     .config("spark.driver.memory", "8g")     .config("spark.executor.memory", "8g")     .config("spark.master", "local[*]")     .getOrCreate()

parquetFile = spark.read.parquet(WIKI_PATH)

# --- PAGERANK CALCULATION ---

def generate_graph(pages):
    edges = pages.flatMap(lambda row: [(row.id, link.id) for link in row.anchor_text]).distinct()
    vertices = pages.flatMap(lambda row: [(row.id,) for link in row.anchor_text] + [(row.id,)]).distinct().map(lambda x: (x[0], x[0]))
    return edges, vertices

pages_links = parquetFile.select("id", "anchor_text").rdd
edges_rdd, vertices_rdd = generate_graph(pages_links)

edgesDF = edges_rdd.toDF(["src", "dst"]).repartition(124, "src")
verticesDF = vertices_rdd.toDF(["id", "title"]).repartition(124, "id")

g = GraphFrame(verticesDF, edgesDF)

pr_results = g.pageRank(resetProbability=0.15, maxIter=3)

pr_rdd = pr_results.vertices.select("id", "pagerank").rdd
pr_dict = pr_rdd.collectAsMap()

with open('pr.pkl', 'wb') as f:
    pickle.dump(pr_dict, f)

os.system(f"gsutil cp pr.pkl gs://{BUCKET_NAME}/pr.pkl")
print("PageRank finished successfully.")

# --- PAGEVIEWS ---
pv_df = spark.read.text(PV_PATH)

def parse_page_views(line):
    parts = line.value.split(" ")
    try:
        page_id = int(parts[0])
        views = int(parts[2])
        return (page_id, views)
    except:
        return None

pv_rdd = pv_df.rdd.map(parse_page_views).filter(lambda x: x is not None)
pv_dict = pv_rdd.collectAsMap()

with open('pageviews.pkl', 'wb') as f:
    pickle.dump(pv_dict, f)

os.system(f"gsutil cp pageviews.pkl gs://{BUCKET_NAME}/pageviews.pkl")
print("PageViews finished.")

# --- TITLE MAPPING ---
print("Starting ID to Title mapping...")
def create_id_to_title_mapping():
    df = parquetFile.select("id", "title")
    id_to_title = df.rdd.collectAsMap()

    with open('id_to_title.pkl', 'wb') as f:
        pickle.dump(id_to_title, f)

    os.system(f"gsutil cp id_to_title.pkl gs://{BUCKET_NAME}/id_to_title.pkl")

create_id_to_title_mapping()


# In[ ]:


# PAGE VIEWS

import os
import pickle

pv_path = 'gs://irprojectbucket/pageviews-202108-user.bz2'
BUCKET_NAME = 'irprojectbucket'

pv_df = spark.read.text(pv_path)

def parse_page_views(line):
    parts = line.value.split(" ")
    try:
        page_id = int(parts[0])
        views = int(parts[2])
        return (page_id, views)
    except:
        return None

pv_rdd = pv_df.rdd.map(parse_page_views).filter(lambda x: x is not None)
pv_dict = pv_rdd.collectAsMap()

with open('pageviews.pkl', 'wb') as f:
    pickle.dump(pv_dict, f)

dest_path = f'gs://{BUCKET_NAME}/pageviews.pkl'
os.system(f"gsutil cp pageviews.pkl {dest_path}")


# In[ ]:


def create_id_to_title_mapping():
    full_path = f"gs://{BUCKET_NAME}/*"
    df = spark.read.parquet(full_path).select("id", "title")

    id_to_title = df.rdd.collectAsMap()

    with open('id_to_title.pkl', 'wb') as f:
        pickle.dump(id_to_title, f)

    client = storage.Client(PROJECT_ID)
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob('id_to_title.pkl')
    blob.upload_from_filename('id_to_title.pkl')

create_id_to_title_mapping()

