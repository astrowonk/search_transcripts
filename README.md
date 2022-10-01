# Transcript Search

This code was designed to make OpenAI's whisper transcripts easily searchable. i.e. to find out at what time in a transcript a certain topic was discusesd, etc. However it should work with any folder of .VTT files.

This module has two classes:

* `LoadTranscripts`: This creates a sqlite database and [a BM 25 index](https://pypi.org/project/rank-bm25/) from a folder of transcript files (`.vtt` or `.json` files). It creates longer chunks of text from the short transcript segments in the original file in order to make the text blocks searchable.

* `SearchTranscripts`: This uses the index and Sqlite database to return a pandas dataframe of the top results for the search query.


Usage:

```{python}

from search_transcripts import LoadTranscripts, SearchTranscripts

l = LoadTranscripts('transcripts') ## will create main.db and bm25.pickle


s = SearchTranscripts()

## uses the bm25 chunked index to score and returns a pandas dataframe of the top scoring transcript sections, across all transcripts.

s.search("starship enterprise") 

## Uses a bm25 index on the entire transcripts to find the transcripts with the highest score for the search term.
s.search_full_transcript("starship enterprise)


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
