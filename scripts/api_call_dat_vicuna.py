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

from llama_cpp import Llama
import random

# download the model file from https://huggingface.co/TheBloke/gpt4-x-vicuna-13B-GGML/resolve/main/gpt4-x-vicuna-13B.ggml.q8_0.bin before running this
llm = Llama(model_path="models/gpt4-x-vicuna-13B-GGML/gpt4-x-vicuna-13B.ggml.q8_0.bin", use_mmap=False, seed=random.randint(0, 1000000))

def generate_response(dat_prompt, temp):
    prompt = f"### Instruction: {dat_prompt}\n### Response: "
    output = llm(prompt, max_tokens=250, temperature=temp)
    llm.reset()
    return output["choices"][0]["text"].strip()


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
    for iterat in range(0, 1000):
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
