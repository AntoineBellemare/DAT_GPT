import os
from openai import OpenAI
import time
import json
import click
import logging
import warnings
warnings.filterwarnings('ignore')
NOTHING="Make a list of 10 words"
YOUNG_NO="Please enter 10 words that are as different from each other as possible, in all meanings and uses of the words. Answer from the perspective of a child. Rules: Only single words in English. Only nouns (e.g., things, objects, concepts). No proper nouns (e.g., no specific people or places). No specialised vocabulary (e.g., no technical terms). Think of the words on your own (e.g., do not just look at objects in your surroundings).  Make a list of these 10 words, a single word in each entry of the list."
NO_STRATEGY="Please enter 10 words that are as different from each other as possible, in all meanings and uses of the words. Rules: Only single words in English. Only nouns (e.g., things, objects, concepts). No proper nouns (e.g., no specific people or places). No specialised vocabulary (e.g., no technical terms). Think of the words on your own (e.g., do not just look at objects in your surroundings).  Make a list of these 10 words, a single word in each entry of the list."

STRATEGY_THE="Please enter 10 words that are as different from each other as possible, in all meanings and uses of the words, using a strategy that relies on using a thesaurus. Rules: Only single words in English. Only nouns (e.g., things, objects, concepts). No proper nouns (e.g., no specific people or places). No specialised vocabulary (e.g., no technical terms). Think of the words on your own (e.g., do not just look at objects in your surroundings). Make a list of these 10 words, a single word in each entry of the list."

STRATEGY_ETYM="Please enter 10 words that are as different from each other as possible, in all meanings and uses of the words, using a strategy that relies on varying etymology. Rules: Only single words in English. Only nouns (e.g., things, objects, concepts). No proper nouns (e.g., no specific people or places). No specialised vocabulary (e.g., no technical terms). Think of the words on your own (e.g., do not just look at objects in your surroundings). Make a list of these 10 words, a single word in each entry of the list."

STRATEGY_OPP="Please enter 10 words that are as different from each other as possible, in all meanings and uses of the words, using a strategy that relies on meaning opposition. Rules: Only single words in English. Only nouns (e.g., things, objects, concepts). No proper nouns (e.g., no specific people or places). No specialised vocabulary (e.g., no technical terms). Think of the words on your own (e.g., do not just look at objects in your surroundings). Make a list of these 10 words, a single word in each entry of the list."

STRATEGY_RAND="Please enter 10 words that are as different from each other as possible, in all meanings and uses of the words, using a strategy that relies on randomness. Rules: Only single words in English. Only nouns (e.g., things, objects, concepts). No proper nouns (e.g., no specific people or places). No specialised vocabulary (e.g., no technical terms). Think of the words on your own (e.g., do not just look at objects in your surroundings). Make a list of these 10 words, a single word in each entry of the list."
strategies = {"nothing":NOTHING,
              "young":YOUNG_NO,
              "none":NO_STRATEGY,
              "etymology":STRATEGY_ETYM,
              "random":STRATEGY_RAND,
              "opposites":STRATEGY_OPP,
              "thesaurus":STRATEGY_THE}
# keys
client = OpenAI()
def generate_response(text, temp):

    completion = client.chat.completions.create(
        model="o4-mini-2025-04-16",
        reasoning_effort="medium",
        messages=[{"role": "user", "content": text}],
        temperature=temp
    )
    return completion.choices[0].message.content.strip()

@click.command()
@click.argument("filename", type=str)
@click.option("--file_path", type=str)
@click.option("--strategy", type=str)
@click.option("--temp", type=float)
@click.option("--iter_nb", type=str)
def main(filename, file_path="./", strategy='none',temp=None, iter_nb='0'):
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
        logger.info(f"API CALL NUMBER {iterat} \n{'~'*80}")
        try:
            response = generate_response(strategies[strategy], temp)
            logger.info(f"Response: \n{response}")
            output.update({iterat:response})
            with open(f"{file_path}{filename}_temp{temp}_{strategy}{iter_nb}.json", "w") as outfile:
                json.dump(output, outfile)
        except:
            logger.info(f"API CALL NUMBER {iterat} FAILED; waiting 30s\n{'~'*80}")
            time.sleep(30)
            continue
        time.sleep(1)
    logger.info(f"done \n {'-'*80}")

if __name__ == "__main__":
    # NOTE: from command line ``
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()