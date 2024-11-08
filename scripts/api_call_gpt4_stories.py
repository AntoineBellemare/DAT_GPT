import os
import openai
import time
import json
import click
import logging
import warnings
warnings.filterwarnings('ignore')
SYNOPSIS="Invent the synopsis of a movie. Be the most original and innovative you can be. Provide an answer of 50 words maximum"
PLOT="Invent the plot of a movie. Be the most creative you can be. Provide an answer of 50 words maximum"
POEM = "Invent a poem. Be the most creative you can be. Provide an answer of 50 words maximum"
HAIKU = "Invent a haiku. Be the most creative you can be."
FLASH_FICTION = "Invent a flash fiction. Be the most creative you can be. Provide an answer of 200 words maximum"
FRENCH_POETIC_FLASH_FICTION = "Écrivez un flash fiction poétique cryptique. Soyez le plus créatif possible. Fournissez une réponse de 200 mots maximum"
FABLE = "Invent a fable. Be the most creative you can be. Provide an answer of 200 words maximum"

SYNOPSIS_nocrea="Invent the synopsis of a movie. Provide an answer of 50 words maximum"
PLOT_nocrea="Invent the plot of a movie. Provide an answer of 50 words maximum"
POEM_nocrea = "Invent a poem. Provide an answer of 8 lines maximum"
HAIKU_nocrea = "Invent a haiku."
FLASH_FICTION_nocrea = "Invent a flash fiction. Provide an answer of 200 words maximum"
FABLE_nocrea = "Invent a fable. Provide an answer of 200 words maximum"

strategies = {"synopsis":SYNOPSIS,
              "plot":PLOT,
              "poem":POEM,
              "haiku":HAIKU, 
              "flash_fiction":FLASH_FICTION,
              "fable":FABLE,
              "synopsis_nocrea":SYNOPSIS_nocrea,
              "plot_nocrea":PLOT_nocrea,
              "poem_nocrea":POEM_nocrea,
              "haiku_nocrea":HAIKU_nocrea,
              "flash_fiction_nocrea":FLASH_FICTION_nocrea,
              "fable_nocrea":FABLE_nocrea,
              "french_poetic_flash_fiction":FRENCH_POETIC_FLASH_FICTION}
# keys
openai.organization = ""
openai.api_key = ""

def generate_response(text, temp):
    response = openai.ChatCompletion.create(model='gpt-4-0314', messages=[{"role":'assistant', "content":text}], temperature=temp)
    #response = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=[{"role":'assistant', "content":text}], temperature=temp)
    return response['choices'][0]["message"]["content"].strip()

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
    for iterat in range(0, int(iter_nb)):
        logger.info(f"API CALL NUMBER {iterat} \n{'~'*80}")
        try:
            response = generate_response(strategies[strategy], temp)
            logger.info(f"Response: \n{response}")
            output.update({iterat:response})
            with open(f"{file_path}{filename}_temp{temp}_{strategy}{iter_nb}.json", "w") as outfile:
                json.dump(output, outfile)
        except:
            logger.info(f"API CALL NUMBER {iterat} FAILED; waiting 1h\n{'~'*80}")
            time.sleep(60)
            continue
        time.sleep(3)
    logger.info(f"done \n {'-'*80}")

if __name__ == "__main__":
    # NOTE: from command line ``
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()