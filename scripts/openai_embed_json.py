import json
import pickle

import numpy as np
import openai
import pandas as pd
from tqdm import trange

# Set the openai.api_key globally
openai.api_key = "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"


def embed(text_json, model="text-embedding-ada-002", chunk_size=500):
    responses = []
    for i in trange(0, len(text_json), chunk_size, desc=f"Embedding chunks"):
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


fname = "GPT3_temp1.0_flash_fiction_nocrea100.json"
# fname = "GPT4_temp1.0_flash_fiction_nocrea100.json"
# fname = "GPT3_temp1.0_haiku_nocrea100.json"
# fname = "GPT4_temp1.0_haiku_nocrea100.json"
# fname = "human_haiku_tempslibres.json"

text_json = json.load(open(f"machine_data_stories/{fname}", "r"))
vecs = embed(text_json)
json.dump(vecs, open(f"machine_data_stories/embeddings/{fname[:-5]}_vecs.json", "w"))