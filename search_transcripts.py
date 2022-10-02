import re
import pandas as pd
import glob
import json
from tqdm.notebook import tqdm
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

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
        self.stop_words = []

    def process_all(self):
        """build search documents and save the database"""
        self.build_search_documents()
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
        self.conn.execute(
            "CREATE VIRTUAL TABLE search_data USING fts5(doc_id, episode_key, text,start, start_segment, end_segment, tokenize = 'porter ascii');"
        )
        df.to_sql('search_data',
                  con=self.conn,
                  if_exists='append',
                  index=False)

        #print(f'Saving {self.output_prefix}bm25.pickle')
        #with open(f'{self.output_prefix}bm25.pickle', 'wb') as f:
        #    pickle.dump(self.bm25, f)

        #self.save_full_transcript_data()

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
        self.search_docs = flatten_list([x[0] for x in self.out])

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
        tokenized_docs = None
        return search_docs, tokenized_docs

    @staticmethod
    def make_timestamp(x):
        """Convert decimal seconds to a string H:M:S.Z timestamp"""
        hours = int(x // 3600)
        minutes = int((x - hours * 3600) // 60)
        seconds = x - hours * 3600 - minutes * 60
        return f"{process_hour(hours)}{minutes:02}:{seconds:05.2f}"


class SearchTranscripts(LoadTranscripts):
    def __init__(self, input_prefix='', con=None):
        """Load the index and connect database"""

        if input_prefix:
            input_prefix = input_prefix + '_'
        # with open(f'{input_prefix}bm25.pickle', 'rb') as f:
        #     self.bm25 = pickle.load(f)
        # with open(f'{input_prefix}bm25_full.pickle', 'rb') as f:
        #     self.bm25_full_transcript = pickle.load(f)
        if isinstance(con, Engine):
            self.conn = con
        else:
            print(f"Using SQL Lite with {input_prefix}main.db ")
            self.conn = create_engine(f'sqlite:///{input_prefix}main.db')

    @staticmethod
    def handle_apostrophe(x):
        """Handle apostrophes inside Sqlite FTS5"""
        if "'" in x:
            fixed = re.sub("'", "''", x)
            return '"' + fixed + '"'
        return x

    def safe_search(self, search):
        """Prepare any query with an apostrophe for FTS5"""
        if "'" not in search:
            return search
        return ' '.join([self.handle_apostrophe(x) for x in search.split(' ')])

    def search_bm25_chunk(self, search):
        """Use the BM 25 index to retrieve the top results from sql."""
        #print(','.join(list(top_indices)))

        df = pd.read_sql(
            f"select bm25(search_data) as score, * from search_data where text MATCH '{self.safe_search(search)}' order by bm25(search_data) limit 50;",
            con=self.conn)

        return df

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
        search = search.lower().strip('"')
        base_res['exact_match'] = base_res['text'].apply(
            lambda x: search in x.lower()).astype(int)
        base_res['text'] = base_res['text'].apply(
            lambda x: process_bold(x, search))

        return base_res.sort_values(['exact_match', 'score'],
                                    ascending=[False, True])


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