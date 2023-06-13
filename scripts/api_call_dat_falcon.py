import requests
import time
import json
import click
import logging
import warnings
NOTHING = "Make a list of 10 words. A single word in each entry of the list."
YOUNG_NO = "Please enter 10 words that are as different from each other as possible, in all meanings and uses of the words. Answer from the perspective of a child. Rules: Only single words in English. Only nouns (e.g., things, objects, concepts). No proper nouns (e.g., no specific people or places). No specialised vocabulary (e.g., no technical terms). Think of the words on your own (e.g., do not just look at objects in your surroundings).  Make a list of these 10 words, a single word in each entry of the list."
NO_STRATEGY = "Please enter 10 words that are as different from each other as possible, in all meanings and uses of the words. Rules: Only single words in English. Only nouns (e.g., things, objects, concepts). No proper nouns (e.g., no specific people or places). No specialised vocabulary (e.g., no technical terms). Think of the words on your own (e.g., do not just look at objects in your surroundings).  Make a list of these 10 words, a single word in each entry of the list."

STRATEGY_THE = "Please enter 10 words that are as different from each other as possible, in all meanings and uses of the words, using a strategy that relies on using a thesaurus. Rules: Only single words in English. Only nouns (e.g., things, objects, concepts). No proper nouns (e.g., no specific people or places). No specialised vocabulary (e.g., no technical terms). Think of the words on your own (e.g., do not just look at objects in your surroundings). Make a list of these 10 words, a single word in each entry of the list."

STRATEGY_ETYM = "Please enter 10 words that are as different from each other as possible, in all meanings and uses of the words, using a strategy that relies on varying etymology. Rules: Only single words in English. Only nouns (e.g., things, objects, concepts). No proper nouns (e.g., no specific people or places). No specialised vocabulary (e.g., no technical terms). Think of the words on your own (e.g., do not just look at objects in your surroundings). Make a list of these 10 words, a single word in each entry of the list."

STRATEGY_OPP = "Please enter 10 words that are as different from each other as possible, in all meanings and uses of the words, using a strategy that relies on meaning opposition. Rules: Only single words in English. Only nouns (e.g., things, objects, concepts). No proper nouns (e.g., no specific people or places). No specialised vocabulary (e.g., no technical terms). Think of the words on your own (e.g., do not just look at objects in your surroundings). Make a list of these 10 words, a single word in each entry of the list."

STRATEGY_RAND = "Please enter 10 words that are as different from each other as possible, in all meanings and uses of the words, using a strategy that relies on randomness. Rules: Only single words in English. Only nouns (e.g., things, objects, concepts). No proper nouns (e.g., no specific people or places). No specialised vocabulary (e.g., no technical terms). Think of the words on your own (e.g., do not just look at objects in your surroundings). Make a list of these 10 words, a single word in each entry of the list."
strategies = {
    "nothing": NOTHING,
    "young": YOUNG_NO,
    "none": NO_STRATEGY,
    "etymology": STRATEGY_ETYM,
    "random": STRATEGY_RAND,
    "opposites": STRATEGY_OPP,
    "thesaurus": STRATEGY_THE,
}

from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

model_name = "tiiuae/falcon-40b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="models/")
model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir="models/", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)

def generate_response(dat_prompt, temp):
    prompt = f">>QUESTION<<{dat_prompt}>>ANSWER<<"
    sequences = pipeline(
        prompt,
        max_length=100,
        do_sample=True,
        temperature=temp,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    return sequences[0]["generated_text"].split(">>ANSWER<<")[1].split(">>QUESTION<<")[0]


@click.command()
@click.argument("filename", type=str)
@click.option("--file_path", type=str)
@click.option("--strategy", type=str)
@click.option("--temp", type=float)
@click.option("--iter_nb", type=str)
def main(filename, file_path="./", strategy="none", temp=None, iter_nb="0"):
    """
    filename: str
        name without , don't forget to change for each new iteration otherwise it will overwrite
        use convention : sample_*strategy*idx (idx being the number of repetition)
    file_path: path, optional
        path to save the file, change if you dont want to install from where you call the script
    strategy:
        choose from ['none','thesaurus','random','etymology', 'opposites']
    
    """
    logger = logging.getLogger(__name__)
    output = {}
    for iterat in range(0, 500):
        logger.info(f"API CALL NUMBER {iterat} \n {'~'*80}")
        try:
            response = generate_response(strategies[strategy], temp)
            logger.info(f"Response: \n{response}")
        except Exception as e:
            logger.info(f"Response: \n{response}")
            logger.info(f"API CALL NUMBER {iterat} FAILED; waiting 1h\n{'~'*80};\nerror: {e}")
            time.sleep(3600)
            continue
        
        output.update({iterat: response})
        with open(
            f"{file_path}{filename}_temp{temp}_{strategy}{iter_nb}.json", "w"
        ) as outfile:
            json.dump(output, outfile)

        time.sleep(1)
    logger.info(f"done \n {'-'*80}")


if __name__ == "__main__":
    # NOTE: from command line ``
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
