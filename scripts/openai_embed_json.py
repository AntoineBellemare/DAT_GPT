import glob
import json
import os
import pickle
from os import path

import numpy as np
import openai
import pandas as pd
from tqdm import tqdm

# Set the openai.api_key globally
openai.api_key = "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"


def embed(text_json, model="text-embedding-ada-002", chunk_size=500):
    if isinstance(text_json, list):
        text_json = {str(i): text for i, text in enumerate(text_json)}

    responses = []
    for i in range(0, len(text_json), chunk_size):
        chunk = list(text_json.values())[i : i + chunk_size]
        if isinstance(chunk[0], dict):
            chunk = [c["overview"] for c in chunk]
        responses.append(openai.Embedding.create(model=model, input=chunk))

    response = []
    for r in responses:
        response += r["data"]
    embeddings = {key: r["embedding"] for key, r in zip(text_json.keys(), response)}
    return embeddings


def embed_all_stories(basedir="machine_data_stories/"):
    if not path.exists(path.join(basedir, "embeddings")):
        os.mkdir(path.join(basedir, "embeddings"))

    fnames = glob.glob(f"{basedir}/*.json")

    pbar = tqdm(fnames, desc="Embedding stories")
    for fname in pbar:
        pbar.set_postfix_str(path.basename(fname))
        fname_vec = f"{basedir}/embeddings/{path.basename(fname)[:-5]}_vecs.json"
        fname_vec = path.normpath(fname_vec)

        if path.exists(fname_vec):
            continue

        with open(fname, "r") as f:
            text_json = json.load(f)
        vecs = embed(text_json)
        with open(fname_vec, "w") as f:
            json.dump(vecs, f)


embed_all_stories()
embed_all_stories(basedir="machine_data_stories/final/")

# embed the word "nature" to analyze the theme of Haikus
print("Embedding the word 'nature'", end="...")
nature_vec = embed(["nature"])["0"]
with open("machine_data_stories/final/embeddings/nature_vec.json", "w") as f:
    json.dump(nature_vec, f)
print("done")
