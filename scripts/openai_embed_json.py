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
    responses = []
    for i in range(0, len(text_json), chunk_size):
        responses.append(
            openai.Embedding.create(
                model=model,
                input=list(text_json.values())[i : i + chunk_size],
            )
        )

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
