from Bard import Chatbot 
import time
import json
import click
import logging
import warnings
import random

warnings.filterwarnings('ignore')
NOTHING="Make a list of 10 words"
NO_STRATEGY="Please enter only 10 words that are as different from each other as possible, in all meanings and uses of the words. Rules: Only single words in English. Only nouns (e.g., things, objects, concepts). No proper nouns (e.g., no specific people or places). No specialised vocabulary (e.g., no technical terms). Think of the words on your own (e.g., do not just look at objects in your surroundings).  Make a list of these 10 words, a single word in each entry of the list. Do not explain your answer."

STRATEGY_THE="Please enter 10 words that are as different from each other as possible, in all meanings and uses of the words, using a strategy that relies on using a thesaurus. Rules: Only single words in English. Only nouns (e.g., things, objects, concepts). No proper nouns (e.g., no specific people or places). No specialised vocabulary (e.g., no technical terms). Think of the words on your own (e.g., do not just look at objects in your surroundings). Make a list of these 10 words, a single word in each entry of the list."

STRATEGY_ETYM="Please enter 10 words that are as different from each other as possible, in all meanings and uses of the words, using a strategy that relies on varying etymology. Rules: Only single words in English. Only nouns (e.g., things, objects, concepts). No proper nouns (e.g., no specific people or places). No specialised vocabulary (e.g., no technical terms). Think of the words on your own (e.g., do not just look at objects in your surroundings). Make a list of these 10 words, a single word in each entry of the list."

STRATEGY_OPP="Please enter 10 words that are as different from each other as possible, in all meanings and uses of the words, using a strategy that relies on meaning opposition. Rules: Only single words in English. Only nouns (e.g., things, objects, concepts). No proper nouns (e.g., no specific people or places). No specialised vocabulary (e.g., no technical terms). Think of the words on your own (e.g., do not just look at objects in your surroundings). Make a list of these 10 words, a single word in each entry of the list."

STRATEGY_RAND="Please enter 10 words that are as different from each other as possible, in all meanings and uses of the words, using a strategy that relies on randomness. Rules: Only single words in English. Only nouns (e.g., things, objects, concepts). No proper nouns (e.g., no specific people or places). No specialised vocabulary (e.g., no technical terms). Think of the words on your own (e.g., do not just look at objects in your surroundings). Make a list of these 10 words, a single word in each entry of the list."
strategies = {"nothing":NOTHING,
              "none":NO_STRATEGY,
              "etymology":STRATEGY_ETYM,
              "random":STRATEGY_RAND,
              "opposites":STRATEGY_OPP,
              "thesaurus":STRATEGY_THE}
token = "WQhWMhpmuVqhBMKbfACNpelyh_7uRCfrxulBxgcI5oyO72j6YWSzDPQDEJBTMNWdwC23eg."
chatbot = Chatbot(token)

def generate_response(text):
    response = chatbot.ask(text)
    return response['content']

@click.command()
@click.argument("filename", type=str)
@click.option("--file_path", type=str)
@click.option("--strategy", type=str)
@click.option("--iter_nb", type=str)
def main(filename, file_path="./", strategy='none', iter_nb='0'):
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
    for iterat in range(0, 5000):
        logger.info(f"API CALL NUMBER {iterat} \n{'~'*80}")
        try:
            response = generate_response(strategies[strategy])
            if "Google Bard encountered an error" in response:
                logger.info(f"API CALL NUMBER {iterat} FAILED; waiting 1h\n{'~'*80}")
                time.sleep(random.randint(3600, 7200))
                continue
            logger.info(f"Response: \n{response}")
            output.update({iterat:response})
            with open(f"{file_path}{filename}_{strategy}{iter_nb}.json", "w") as outfile:
                json.dump(output, outfile)
        except:
            logger.info(f"API CALL NUMBER {iterat} FAILED; waiting 1h\n{'~'*80}\n{generate_response(strategies[strategy])}")
            time.sleep(3600)
            continue
        time.sleep(random.randint(5,20))
    logger.info(f"done \n {'-'*80}")

if __name__ == "__main__":
    # NOTE: from command line `unofficial_api filename`
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()