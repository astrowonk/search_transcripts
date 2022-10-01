# Transcript Search

This code was designed to make OpenAI's whisper transcripts easily searchable. i.e. to find out at what time in a transcript a certain topic was discusesd, etc.

This module has two classes:

* `LoadTranscripts`: This creates a sqlite database and a BM 25 index from a a folder of transcript files (`.vtt` or `.json` files). It creates longer chunks of text from the short transcript segments in the original file in order to make the text blocks searchable.

* `SearchTranscripts`: This uses the index and Sqlite database to return a pandas dataframe of the top results for the search query.


Usage:

```{python}

from search_transcripts import LoadTranscripts, SearchTranscripts

l = LoadTranscripts('transcripts') ## will create main.db and bm25.pickle



s = SearchTranscripts()


```