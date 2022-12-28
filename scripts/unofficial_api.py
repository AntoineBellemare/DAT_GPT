from pyChatGPT import ChatGPT
import json
import time
import click
import logging
import
# either make a request with email/pwd or with session token
# check instructions https://github.com/terry3041/pyChatGPT#obtaining-session_token
# BEST WORKFLOW : first login using email/pwd, solve captcha manually
# then other api calls will use session tokens solving captchas automatically
OPENAI_EMAIL=""
OPENAI_PASSWORD=""

SESSION_TOKEN1 = ""
SESSION_TOKEN2 = ""

NO_STRATEGY="Please enter 10 words that are as different from each other as possible, in all meanings and uses of the words. Rules: Only single words in English. Only nouns (e.g., things, objects, concepts). No proper nouns (e.g., no specific people or places). No specialised vocabulary (e.g., no technical terms). Think of the words on your own (e.g., do not just look at objects in your surroundings).  Make a list of these 10 words, a single word in each entry of the list."

STRATEGY_THE="Please enter 10 words that are as different from each other as possible, in all meanings and uses of the words, using a strategy that relies on using a thesaurus. Rules: Only single words in English. Only nouns (e.g., things, objects, concepts). No proper nouns (e.g., no specific people or places). No specialised vocabulary (e.g., no technical terms). Think of the words on your own (e.g., do not just look at objects in your surroundings). Make a list of these 10 words, a single word in each entry of the list."

STRATEGY_ETYM="Please enter 10 words that are as different from each other as possible, in all meanings and uses of the words, using a strategy that relies on varying etymology. Rules: Only single words in English. Only nouns (e.g., things, objects, concepts). No proper nouns (e.g., no specific people or places). No specialised vocabulary (e.g., no technical terms). Think of the words on your own (e.g., do not just look at objects in your surroundings). Make a list of these 10 words, a single word in each entry of the list."

STRATEGY_OPP="Please enter 10 words that are as different from each other as possible, in all meanings and uses of the words, using a strategy that relies on meaning opposition. Rules: Only single words in English. Only nouns (e.g., things, objects, concepts). No proper nouns (e.g., no specific people or places). No specialised vocabulary (e.g., no technical terms). Think of the words on your own (e.g., do not just look at objects in your surroundings). Make a list of these 10 words, a single word in each entry of the list."

STRATEGY_RAND="Please enter 10 words that are as different from each other as possible, in all meanings and uses of the words, using a strategy that relies on randomness. Rules: Only single words in English. Only nouns (e.g., things, objects, concepts). No proper nouns (e.g., no specific people or places). No specialised vocabulary (e.g., no technical terms). Think of the words on your own (e.g., do not just look at objects in your surroundings). Make a list of these 10 words, a single word in each entry of the list."

SESSION_TOKENS = [SESSION_TOKEN1, SESSION_TOKEN2]
# check instruction link above; have to create account to auto-solve captcha
twocaptcha_token = ""
strategies = {'none': NO_STRATEGY, 'thes':STRATEGY_THE, 'opp': STRATEGY_OPP, 'rand':STRATEGY_RAND, 'etym':STRATEGY_ETYM}

def _session_token_switch(n, tokens = SESSION_TOKENS):
    """
    internal function to switch accounts
    n : counter

    returns either token
    """
    return tokens[int(n%2)]

@click.command()
@click.argument("filename", type=str)
@click.option("--output_filepath", type=click.Path())
@click.option("--strategy", type=str)
def main(filename, file_path="./", strategy='none'):
    """
    filename: str
        name without , don't forget to change for each new iteration otherwise it will overwrite
        use convention : sample_*strategy*idx (idx being the number of repetition)
    file_path: path, optional
        path to save the file, change if you dont want to install from where you call the script
    strategy:
        choose from ['none','thes','rand','etym', 'opp']
    """
    logger = logging.getLogger(__name__)
    # init dict to save answers
    output = {}
    # intial API call - choose from either mode; if auth_type is email, have to solve captcha manually
    api = ChatGPT(auth_type='openai', email=OPENAI_EMAIL, password=OPENAI_PASSWORD,
                 verbose=True, moderation=False,login_cookies_path="~/AppData/Local/Google/Chrome/User\ Data/Default")
    #api = ChatGPT(SESSION_TOKENS[0],verbose=True, moderation=False,
    #            login_cookies_path="~/AppData/Local/Google/Chrome/User\ Data/Default",
    #            twocaptcha_apikey=twocaptcha_token)
    # not elegant but serve us to switch accounts
    counter = 0
    for samples in range(500):
        logger.info(f"API CALL NUMBER : {samples+1} \n")
        if samples >= 1:
            api = ChatGPT(SESSION_TOKENS[0], moderation=False,
                          twocaptcha_apikey=twocaptcha_token,
                          login_cookies_path="~/AppData/Local/Google/Chrome/User\ Data/Default",
                          verbose=True)
        try:
            resp = api.send_message(strategies[strategy])
        # it catches any error and interprets it like failing api call, so switch accounts
        # it still crashes internally with list index out of range - can't do anything about it
        except:
            logger.info('ERROR: SWITCHING ACCOUNTS \n')
            api.driver.quit()
            counter +=1
            time.sleep(63)
            api = ChatGPT(_session_token_switch(n=counter), moderation=False,
                        twocaptcha_apikey=twocaptcha_token,
                        login_cookies_path="~/AppData/Local/Google/Chrome/User\ Data/Default",
                        verbose=True)
        try:    
            resp = api.send_message(strategies[strategy])
        except:
            api.driver.quit()
            api = ChatGPT(auth_type='openai', email=OPENAI_EMAIL, password=OPENAI_PASSWORD,
                          verbose=True, moderation=False,
                          login_cookies_path="~/AppData/Local/Google/Chrome/User\ Data/Default")
            resp = api.send_message(strategies[strategy])
        # Save whole dictionary of outputs everytime add a response
        output.update({samples:resp['message']})
        with open(f"{file_path}/{filename}.json", "w") as outfile:
            json.dump(output, outfile)
        
        # reset and refresh
        api.reset_conversation()
        api.refresh_chat_page()
        time.sleep(63)
    logger.info("done")

if __name__ == "__main__":
    # NOTE: from command line `unofficial_api filename`
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()