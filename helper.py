import json
from os import path

import whisper
import datetime
import pycurl


def download_url(x):
    c = pycurl.Curl()
    c.setopt(c.URL, x)
    c.setopt(c.FOLLOWLOCATION, True)
    with open(x.split("/")[-1], 'wb') as f:
        c.setopt(c.WRITEFUNCTION, f.write)
        c.perform()


def write_json(out, filename):

    with open(filename, 'w') as f:
        json.dump([{
            'start': x['start'],
            'end': x['end'],
            'text': x['text']
        } for x in out['segments']],
                  f,
                  indent=4)


def transcribe_to_json(filename, model='base.en'):
    out_name = f'{filename}_transcript_{model}.json'
    if path.exists(out_name):
        print(f"File {out_name} exists")
        return

    start_time = datetime.datetime.now()
    model = whisper.load_model(model)

    out = model.transcribe(filename, language='en', verbose=False)
    end_time = datetime.datetime.now()
    print(f"This took {(end_time-start_time).total_seconds() / 60} minutes")
    write_json(out, out_name)


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('filename')
    parser.add_argument("--model", default='medium.en')

    args = parser.parse_args()

    transcribe_to_json(args.filename, model=args.model)
