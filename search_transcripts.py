from email.mime import base
import re
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from rank_bm25 import BM25Okapi
import numpy as np
import pandas as pd
import glob
import json
from tqdm.notebook import tqdm
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
import pickle

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count


def flatten_list(list_of_lists):
    return [y for x in list_of_lists for y in x]


class LoadTranscripts():
    """Load a directory of VTT or .json transcripts (from Whisper) into a sqlite database. It also creates an BM25 index.
    
    This creates the data for SearchTranscripts()
    
    """
    search_docs = None
    chunk_length = 30
    conn = None

    def __init__(self,
                 path: str,
                 chunk_length: int = 25,
                 overlap_length: int = 0,
                 key_regex: str = None,
                 output_prefix: str = '',
                 con: Engine = None) -> None:
        """Initalize the class. path will be globbed for .vtt and .json files.
        
        the episode_key will come from the filename unless key_regex is specified to extract an episode number or other identifier.
        
        """
        if con:
            self.conn = con
            assert isinstance(con, Engine), f"{con} is not a SQLalchemy engine"
        if output_prefix:
            output_prefix = output_prefix + '_'
        self.output_prefix = output_prefix
        if chunk_length:
            assert isinstance(chunk_length,
                              int), "Chunk length must be an integer."
            self.chunk_length = chunk_length
        self.overlap_length = overlap_length
        self.key_regex = key_regex
        self.load_all_files(path)
        self.porter_stemmer = PorterStemmer()
        self.stop_words = []

    def process_all(self):
        """build search documents and save the database"""
        self.build_search_documents()
        self.build_full_transcript_search_index()
        self.save_data()

    def load_all_files(self, path):
        """Load all files into self.data, a list of dictionaries."""
        json_files = glob.glob(f"{path}/*.json")
        vtt_files = glob.glob(f"{path}/*.vtt")
        print('loading data')
        if not self.key_regex:
            self.data = {x: json.load(open(x)) for x in tqdm(json_files)}
            self.data.update({x: read_vtt(x) for x in tqdm(vtt_files)})
        else:  #TODO: need to handle .VTT files sadly.
            self.data = {
                self.process_regex(x): json.load(open(x))
                for x in tqdm(json_files)
            }
            self.data.update(
                {self.process_regex(x): read_vtt(x)
                 for x in tqdm(vtt_files)})

    def process_regex(self, x):
        """turn the file names into whatever the regex finds.  Regex should have a () group."""
        print(x)
        m = re.search(self.key_regex, x)
        return m.group(1)

    def save_data(self):
        """take the list of dictionaries and create the sqlite table and indices."""
        if self.conn is None:
            self.conn = create_engine(f'sqlite:///{self.output_prefix}main.db')
        assert isinstance(self.conn,
                          Engine), "Connection must be a sqlachemy engine."
        print(f"Writing SQL with {self.conn}")

        print("Making table all_segments")
        self.conn.execute(
            "drop table if exists all_segments;"
        )  #this probably doesn't work on a bunch of other connection types
        # TODO add warning about dropping table

        ## segment data
        for key in self.data.keys():
            df = pd.json_normalize(
                self.data[key]).reset_index().rename(columns={
                    'index': 'segment'
                }).drop(columns='end')
            df['start'] = df['start'].apply(self.make_timestamp)
            df['episode_key'] = key
            df.to_sql('all_segments',
                      con=self.conn,
                      if_exists='append',
                      index=False)

        self.conn.execute(
            "CREATE INDEX idx_segments on all_segments(segment,episode_key);")

        ## search chunk data
        print("Making table search_data")

        self.conn.execute("drop table if exists search_data;")
        df = pd.DataFrame(self.search_docs).drop(
            columns=['end']).reset_index().rename(columns={'index': 'doc_id'})
        df.to_sql('search_data',
                  con=self.conn,
                  if_exists='append',
                  index=False)
        print("Making indices")

        self.conn.execute(
            "CREATE UNIQUE INDEX idx_search_doc_id on search_data(doc_id)")

        self.conn.execute(
            "CREATE INDEX idx_search_doc_id_key on search_data(doc_id,episode_key)"
        )

        print(f'Saving {self.output_prefix}bm25.pickle')
        with open(f'{self.output_prefix}bm25.pickle', 'wb') as f:
            pickle.dump(self.bm25, f)

        self.save_full_transcript_data()

    def save_full_transcript_data(self):
        """ save the full transcript index and sql table"""
        self.full_transcript_df.to_sql('full_episodes',
                                       con=self.conn,
                                       if_exists='replace')

        print(f'Saving {self.output_prefix}bm25_full.pickle')
        with open(f'{self.output_prefix}bm25_full.pickle', 'wb') as f:
            pickle.dump(self.bm25_full_transcript, f)

    def build_search_documents(self):
        """tokenize and segment each transcript."""
        self.tokenized_docs = []
        self.search_docs = []

        workers = cpu_count()
        print(f'build search documents with {workers} workers')

        with ProcessPoolExecutor(max_workers=workers) as executor:
            out = list(
                tqdm(executor.map(self.create_rolling_docs, self.data.items()),
                     total=len(self.data)))
        self.out = out
        self.process_search_docs()

    def process_search_docs(self):
        self.search_docs = flatten_list([x[0] for x in self.out])
        self.tokenized_docs = flatten_list([x[1] for x in self.out])
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def build_full_transcript_search_index(self):
        """    make an index on the full transcript
        """
        data_list = []
        for key, data in tqdm(self.data.items()):
            data = "".join([x['text'] for x in data])
            data_list.append(self.stem_text(data))

        self.full_transcript_df = pd.DataFrame(pd.Series(self.data.keys()),
                                               index=range(
                                                   len(self.data.keys())),
                                               columns=['episode_key'])
        self.bm25_full_transcript = BM25Okapi(data_list)

    def create_rolling_docs(self, x):
        """For a given transcript, chunk 30 segments together to make an indexable document."""
        i = 0
        all_chunk = []
        key, data = x
        while i < (data_length := len(data)):
            last_index = min(i + self.chunk_length, data_length - 1)
            chunk = data[i:last_index + 1]

            start_time = chunk[0]['start']
            full_text = ' '.join([x['text'] for x in chunk])

            #url = self.make_url(start_time)

            all_chunk.append({
                'episode_key':
                key,
                'text':
                full_text,
                'start':
                self.make_timestamp(start_time),
                'end':
                self.make_timestamp(chunk[-1]['end']),
                'start_segment':
                i,
                'end_segment':
                min(i + self.chunk_length, data_length)
            })
            i = i + self.chunk_length + self.overlap_length
        search_docs = all_chunk
        tokenized_docs = [self.stem_text(x['text']) for x in search_docs]
        return search_docs, tokenized_docs

    def search(self, search):
        """search the index and return documents from the search_docs attribute. Used mostly for testing the index."""
        res = self.bm25.get_scores(self.stem_text(search))
        top_indices = np.argsort(-res)[:25]
        scores = res[top_indices]
        df = pd.DataFrame(self.search_docs).iloc[top_indices]
        df['score'] = scores
        df['_exact_match'] = df['text'].apply(
            lambda x: search.lower() in x.lower()).astype(int)
        return df.sort_values(['_exact_match', 'score'], ascending=False)

    @staticmethod
    def make_timestamp(x):
        """Convert decimal seconds to a string H:M:S.Z timestamp"""
        hours = int(x // 3600)
        minutes = int((x - hours * 3600) // 60)
        seconds = x - hours * 3600 - minutes * 60
        return f"{process_hour(hours)}{minutes:02}:{seconds:05.2f}"

    def stem_text(self, text):
        """Stem with word tokenize and the porter stemmer"""
        text = re.sub(r'\W+', ' ', text)
        tokens = [
            w.lower() for w in word_tokenize(text)
            if not w.lower() in self.stop_words
        ]
        return [self.porter_stemmer.stem(item.lower()) for item in tokens]


class SearchTranscripts(LoadTranscripts):
    def __init__(self, input_prefix='', con=None):
        """Load the index and connect database"""

        if input_prefix:
            input_prefix = input_prefix + '_'
        with open(f'{input_prefix}bm25.pickle', 'rb') as f:
            self.bm25 = pickle.load(f)
        with open(f'{input_prefix}bm25_full.pickle', 'rb') as f:
            self.bm25_full_transcript = pickle.load(f)
        if isinstance(con, Engine):
            self.conn = con
        else:
            print(f"Using SQL Lite with {input_prefix}main.db ")
            self.conn = create_engine(f'sqlite:///{input_prefix}main.db')
        self.porter_stemmer = PorterStemmer()
        self.stop_words = []

    def search_bm25_chunk(self, search):
        """Use the BM 25 index to retrieve the top results from sql."""
        res = self.bm25.get_scores(self.stem_text(search))
        top_indices = np.argsort(-res)[:25]
        scores = res[top_indices]
        score_map = {key: val for key, val in zip(top_indices, scores)}
        #print(','.join(list(top_indices)))
        df = pd.read_sql(
            f"select * from search_data where doc_id in ({','.join([str(x) for x in top_indices])})",
            con=self.conn)

        df['score'] = df['doc_id'].map(score_map)
        return df.query('score > 0').sort_values('score', ascending=False)

    def get_segment_detail(self, key, start, end):
        """Get the text of the appropriate segments from sql. a future version may create time stamp specicifc links for each section."""
        return pd.read_sql(
            f"SELECT * from all_segments where episode_key = '{key}' and segment BETWEEN {start} and {end}",
            con=self.conn)

    def assemble_chunk_text(self, start, end, key):

        segment_details = self.get_segment_detail(key, start,
                                                  end).to_dict('records')
        return ''.join([x['text'] for x in segment_details])

    def search(self, search):
        base_res = self.search_bm25_chunk(search)
        base_res['text'] = base_res[[
            'start_segment', 'end_segment', 'episode_key'
        ]].apply(lambda x: self.assemble_chunk_text(x[0], x[1], x[2]), axis=1)
        base_res['exact_match'] = base_res['text'].apply(
            lambda x: search.lower() in x.lower()).astype(int)
        base_res['text'] = base_res['text'].apply(
            lambda x: process_bold(x, search))

        return base_res.sort_values('exact_match', ascending=False)

    def exact_string_search(self, search, limit_with_index=False):
        """Query text blocks directly with sql lite, optionally limiting scope using the Bm25 index."""
        if not limit_with_index:
            query = f"select * from search_data where text like '%{search}%'"
            base_res = pd.read_sql(query, con=self.conn)

        else:
            scores = self.bm25.get_scores(self.stem_text(search))
            nonzerolocs = np.where(scores > 0)[0]
            nonzeroscores = [str(x) for x in nonzerolocs]
            score_map = pd.Series(scores[nonzerolocs],
                                  index=nonzerolocs,
                                  name='score')
            query = f"select * from (select * from search_data where doc_id in ({','.join([str(x) for x in list(nonzeroscores)])})) where text like '%{search}%';"
            base_res = pd.read_sql(query, con=self.conn)
            base_res['score'] = base_res['doc_id'].map(score_map)

        base_res['exact_match'] = base_res['text'].apply(
            lambda x: search.lower() in x.lower()).astype(int)
        base_res['text'] = base_res['text'].apply(
            lambda x: process_bold(x, search))

        return base_res.sort_values('exact_match', ascending=False)

    def search_full_transcript(self, search):
        res = self.bm25_full_transcript.get_scores(self.stem_text(search))
        top_indices = np.argsort(-res)[:10]
        scores = res[top_indices]
        df = pd.read_sql(
            f'select * from full_episodes where "index" in ({",".join([str(x) for x in top_indices])})',
            con=self.conn)
        df['scores'] = scores
        return df


def process_bold(x, search):
    idx = x.lower().find(search.lower())
    if idx != -1:
        return ''.join([
            x[0:idx], '**', x[idx:idx + len(search)], '**',
            x[idx + len(search):]
        ])
    return x


def convert_timestamp(ts):
    """Convert a timestamp H:M:S.Z into decimal seconds"""
    return sum(float(x) * 60**i for i, x in enumerate(reversed(ts.split(':'))))


def read_vtt(filename):
    """Load a VTT file into a list of dictionaries, similar to the json structure created by the whsiper.trascribe() python API."""

    try:
        out = []
        with open(filename, 'r') as f:
            _ = f.readline()  # skip first two lines
            _ = f.readline()
            for line1 in f:
                if not line1.strip():
                    continue
                while not (line2 := next(f)):
                    pass
                start, end = [x.strip() for x in line1.split("-->")]
                out.append({
                    'start': convert_timestamp(start),
                    'end': convert_timestamp(end),
                    'text': line2.strip('\n'),
                })
    except StopIteration:
        print("stop")
    return out


def process_hour(x):
    """simple conditional for hour format"""
    if x:
        return f"{x:02}:"
    return ''