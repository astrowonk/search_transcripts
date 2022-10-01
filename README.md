# Transcript Search

This module has two classes:

* LoadTranscripts: This loads a folder of transcript files (.vtt or a .json file) into a SQL Lite database and creates a BM25 index on small chunks of the transcript. This allows to find specific time-stamped sections of the transcript.

* SearchTranscripts: This uses the index and Sqlite database to return a pandas dataframe of the top results for the search query.

