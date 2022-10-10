# Transcript Search

This code was designed to make a large number of OpenAI's whisper transcripts easily searchable. So one can find a particular passage or term occurs and at what time in the transcript. However, it should work with any folder of .VTT files: non just OpenAI transcripts of podcasts.

I used Whisper OpenAI to transcribe [the Accidental Tech Podcast](https://atp.fm), and [a live search engine of the transcipts are here](https://marcoshuerta.com/dash/atp_search/), powered by this module (specifically `SearchTranscripts`).

This module has two classes:

* `LoadTranscripts`: This creates a sqlite database and [FTS5 virtual table](https://www.sqlite.org/fts5.html) from a folder of transcript files (`.vtt` or `.json` files). It creates longer chunks of text (about 300 words each) from the short transcript segments in the original file in order to make the text blocks searchable. It preserves the individual transcript segments in a separate database.

* `SearchTranscripts`: This is a python class that uses the Sqlite database to return a pandas dataframe of the top results for the search query.

Once the sqlite database is created with `LoadTranscripts`, you can access that database via any sqlite interface you like such as [datasette](https://datasette.io), [dbeaver](https://dbeaver.io), the [command line](https://www.sqlite.org/cli.html), [SQL alchemy](https://www.sqlalchemy.org), etc. The `SearchTranscripts` class is meant to be a simple and convenient way to access the data from python, using the built-in sqlite3 module and pandas, but it is somewhat limited.

Usage:

```{python}

from search_transcripts import LoadTranscripts, SearchTranscripts

l = LoadTranscripts('transcripts') ## will create main.db and bm25.pickle


s = SearchTranscripts()

## Returns a pandas dataframe of the top scoring transcript sections, across all transcripts.

s.search('starship enterprise')

##find the exact phrase

s.search('"starship enterprise"')


```

## JSON transcripts?

 So, before I realized Whisper would create a standard .VTT file, I was using the python API directly. It generates a list of python dictionaries. Saving that as JSON seemed logical at the time. I find the JSON much more easily machine readable than .VTT, and can easily be converted to VTT, so I still support this somewhat quirky format. It looks like so:

```{json}
    [
           {
        "start": 606.1800000000001,
        "end": 610.74,
        "text": " It's important to have a goal to work toward and accomplish rather than just randomly learning and half building things"
    },
    {
        "start": 610.74,
        "end": 613.0600000000001,
        "text": " Having a specific thing you want to build is a good substitute"
    },
    {
        "start": 613.38,
        "end": 619.78,
        "text": " Keep making things until you've made something you're proud enough a proud of enough to show off in an interview by the time you've built a few"
    },
    {
        "start": 619.78,
        "end": 624.26,
        "text": " Things you'll start developing the taste you need to make that determination of what's quote unquote good enough"
    },


    ]

```
