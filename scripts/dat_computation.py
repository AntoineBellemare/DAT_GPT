"""Compute score for Divergent Association Task,
a quick and simple measure of creativity
(Copyright 2021 Jay Olson; see LICENSE)"""

import re
import itertools
import pandas as pd
import numpy as np
import json
import os
import re
import glob
import warnings
import scipy.spatial.distance
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

colors = {'GPT-3_low_DAT': '#FDB813',
          'GPT-3_mid_DAT':'#EF6C00',
          'GPT-3_high_DAT': '#D32F2F',
          'GPT-3_low_control': '#EEE8AA',
          'GPT-3_mid_control': '#FFE082',
          'GPT-3_high_control':'#FFAB91',
          'GPT-4_low_DAT':'#00B7C3',
          'GPT-4_mid_DAT':'#3F51B5',
          'GPT-4_high_DAT':'#9C27B0',
          'GPT-4_low_control':'#80DEEA',
          'GPT-4_mid_control':'#8C9EFF',
          'GPT-4_high_control':'#CE93D8',
          'Human (8k)':'black',
          'Human (100k)':'darkgrey',
          'GPT-3_mid_ety': '#26A69A',
          'GPT-3_mid_thes': '#D81B60',
          'GPT-3_mid_opp': '#FFD700',  # goldenrod
          'GPT-3_mid_rnd':'#7E57C2',
          'GPT-4_mid_ety': '#1A7466',
          'GPT-4_mid_thes': '#A51645',
          'GPT-4_mid_opp': '#BBA600',
          'GPT-4_mid_rnd':'#4A3280',
          'Bard_mid_DAT':'peru',
          'Bard_mid_control':'#DEC3A3',
          'Claude_low_DAT':'#FDB813',
          'Claude_mid_DAT':'teal',
          'Claude_high_DAT':'#D32F2F',
          'Claude_mid_control':'#1ACCCC',
          'Pythia_mid_DAT':'#603080',
          'Pythia_mid_control':'#BF9FDF',
          'StableLM_mid_DAT':'darkgreen',
          'StableLM_mid_control':'#80FF80',
          'StableLMoass_mid_DAT':'deeppink',
          'StableLMoass_mid_control':'pink',
          'RedPajama_mid_DAT':'#FF0000',
          'RedPajama_mid_control':'#FF8080',
          'Vicuna_mid_DAT':'#7BC8F6',
          'Vicuna_mid_control':'#ADD8E6',
          'Vicuna_mid_ety':'#7BC8F6',}

class Model:
    """Create model to compute DAT"""

    def __init__(self, model="glove.840B.300d.txt", dictionary="words.txt", pattern="^[a-z][a-z-]*[a-z]$"):
        """Join model and words matching pattern in dictionary"""

        # Keep unique words matching pattern from file
        words = set()
        with open(dictionary, "r", encoding="utf8") as f:
            for line in f:
                if re.match(pattern, line):
                    words.add(line.rstrip("\n"))

        # Join words with model
        vectors = {}
        with open(model, "r", encoding="utf8") as f:
            for line in f:
                tokens = line.split(" ")
                word = tokens[0]
                if word in words:
                    vector = np.asarray(tokens[1:], "float32")
                    vectors[word] = vector
        self.vectors = vectors


    def validate(self, word):
        """Clean up word and find best candidate to use"""

        # Strip unwanted characters
        clean = re.sub(r"[^a-zA-Z- ]+", "", word).strip().lower()
        if len(clean) <= 1:
            return None # Word too short

        # Generate candidates for possible compound words
        # "valid" -> ["valid"]
        # "cul de sac" -> ["cul-de-sac", "culdesac"]
        # "top-hat" -> ["top-hat", "tophat"]
        candidates = []
        if " " in clean:
            candidates.append(re.sub(r" +", "-", clean))
            candidates.append(re.sub(r" +", "", clean))
        else:
            candidates.append(clean)
            if "-" in clean:
                candidates.append(re.sub(r"-+", "", clean))
        for cand in candidates:
            if cand in self.vectors:
                return cand # Return first word that is in model
        return None # Could not find valid word


    def distance(self, word1, word2):
        """Compute cosine distance (0 to 2) between two words"""

        return scipy.spatial.distance.cosine(self.vectors.get(word1), self.vectors.get(word2))


    def dat(self, words, minimum=7):
        """Compute DAT score"""
        # Keep only valid unique words
        uniques = []
        for word in words:
            valid = self.validate(word)
            if valid and valid not in uniques:
                uniques.append(valid)
        print('Number of valid words', len(uniques))
        # Keep subset of words
        if len(uniques) >= minimum:
            subset = uniques[:minimum]
        else:
            return None # Not enough valid words

        # Compute distances between each pair of words
        distances = []
        for word1, word2 in itertools.combinations(subset, 2):
            dist = self.distance(word1, word2)
            distances.append(dist)
        self.distances = distances
        # Compute the DAT score (average semantic distance multiplied by 100)
        return (sum(distances) / len(distances)) * 100

def extract_words_with_stars(input_list):
    pattern = r'\*\*(\w+)\*\*'
    words_with_stars = []

    for string in input_list:
        matches = re.findall(pattern, string)
        if matches:
            words_with_stars.extend(matches)

    return words_with_stars


# GloVe model from https://nlp.stanford.edu/projects/glove/
model_dat = Model("../model/glove.840B.300d.txt", "words.txt")

#load DAT Olson's data
filename = "../human_data_dat/study2.tsv"

# read the data into a DataFrame
df_human = pd.read_csv(filename, sep='\t')

# extract the "dat" column as a list of floats
dat_study = df_human['dat'].astype(float).tolist()

#load DAT big data file
global_crea = pd.read_csv('../human_data_dat/global-creativity.csv')
DAT_bigdata = global_crea['score']

#load 100k human data
final_human_data = pd.read_csv('../human_data_dat/ai-creativity.csv')
DAT_100k = final_human_data['score']

# Define the file path where the data is located
data_path = '../machine_data_dat/'

# Define a dictionary to store the results of model.dat(words)
results_dict = {'Temperature': [], 'Strategy': [], 'Score': [], 'Model': [], 'Control': [], 'Words': []}

# keep track of these so that we can apply further methods of words extractions
bard_data = []
pythia_data = []

# define counters
counter_bard = 0
counter_pythia = 0
counter_gpt3 = 0
counter_gpt4 = 0
counter_claude = 0
counter_stablelm = 0
counter_stablelmoasst = 0
counter_red = 0
counter_vicuna = 0

# Loop through each file in the data path
for file in sorted(glob.glob('../machine_data_dat/*.json')):
    # Open the file and load the JSON data
    with open(file, 'r') as f:
        data = json.load(f)
    file = os.path.basename(file)
    # Loop through each key in the JSON data
    for i in data.keys():
        # Split the words into a list
        words = data[i].split()
        if 'bard' in file:
            words = extract_words_with_stars(words)
            bard_data.append(words)
        else:
            # Find the indices of '1.' to '10.'
            indices = [m.start() for m in re.finditer(r'\b(?:[1-9]|10)\.', data[i])]  

            # Extract the words after '1.' to '10.' and store them in a new list
            new_words = []
            for idx in range(len(indices)):
                if idx < len(indices) - 1:
                    new_words.append(data[i][indices[idx] + 3:indices[idx + 1]].strip())
                else:
                    new_words.append(data[i][indices[idx] + 3:].strip())

            words = new_words
        
        # Define the strategy based on the file name
        if 'sample_thes' in file or 'gpt4_thes' in file:
            strategy = 'Thesaurus'
        elif 'sample_oppo' in file or 'gpt4_oppo' in file:
            strategy = 'Opposition'
        elif 'ety' in file:
            strategy = 'Etymology'
        elif 'sample_rand' in file:
            strategy = 'Random'
        elif 'none' in file:
            strategy = 'Original instructions'
        elif 'nothing' in file:
            strategy = 'Control'
        else:
            strategy = 'Original instructions'
        # Define the temperature based on the file name
        if 'temp1.5' in file:
            condition = 'High'
        elif 'temp0.5' in file:
            condition = 'Low'
        elif 'temp1.0' in file:
            condition = 'Mid'
        elif 'temp0.7' in file or "temp0.8" in file:
            condition = 'Mid'
        elif 'temp0.2' in file:
            condition = 'Low'
        elif 'temp0.9' in file:
            condition = 'High'
        elif 'temp1.0' and 'pythia' in file:
            condition = 'High'
        elif 'temp1.2' and 'claude' in file:
            condition = 'High'
        else:
            condition = 'Mid'
        
        # Define the model based on the file name
        if 'gpt4' in file:
            llm = 'GPT-4'
            if condition == 'Mid' and strategy == 'Original instructions':
                counter_gpt4 += 1
        elif 'claude' in file:
            llm = 'Claude'
            if condition == 'Mid' and strategy == 'Original instructions':
                counter_claude += 1
        elif 'bard' in file:
            llm = 'Bard'
            if condition == 'Mid' and strategy == 'Original instructions':
                counter_bard += 1
        elif 'pythia' in file:
            llm = 'Pythia'
            if condition == 'Mid' and strategy == 'Original instructions':
                counter_pythia += 1
        elif 'oasst_stablelm' in file:
            llm = 'StableLM-oasst'
            if condition == 'Mid' and strategy == 'Original instructions':
                counter_stablelmoasst += 1
        elif 'stablelm' in file:
            llm = 'StableLM'
            if condition == 'Mid' and strategy == 'Original instructions':
                counter_stablelm += 1
        elif 'redpajama7B' in file:
            llm='RedPajama'
            if condition == 'Mid' and strategy == 'Original instructions':
                counter_red += 1
        elif 'redpajama' in file:
            llm='RedPajama3B'
            #if condition == 'Mid' and strategy == 'Original instructions':
            #    counter_red += 1
        elif 'vicuna' in file:
            llm = 'Vicuna'
            if condition == 'Mid' and strategy == 'Original instructions':
                counter_vicuna += 1
        else:
            llm = 'GPT-3'
            if condition == 'Mid' and strategy == 'Original instructions':
                counter_gpt3 += 1
        
        # Loop through each word in the list
        score = model_dat.dat(words)
            
        # Append the results to the dictionary
        results_dict['Temperature'].append(condition)
        results_dict['Strategy'].append(strategy)
        results_dict['Score'].append(score)
        results_dict['Words'].append(words)
        results_dict['Model'].append(llm)
        # Add a columns with binary Control vs. experimental
        if strategy == 'Control':
            results_dict['Control'].append('Control')
        elif strategy == 'Original instructions':
            results_dict['Control'].append('Original instructions')
        else:
            results_dict['Control'].append('Strategy')

# Convert the results dictionary to a Pandas DataFrame
results_df = pd.DataFrame(results_dict)

# Concatenate the website data with the results DataFrame
results_df = pd.concat([results_df, pd.DataFrame({'Temperature': np.tile(None, len(DAT_bigdata)),
                                                  'Strategy': np.tile('Original instructions', len(DAT_bigdata)),
                                                  'Score': np.array(DAT_bigdata),
                                                  'Model': np.tile('Human (750k)', len(DAT_bigdata)),
                                                  'Control': np.tile('Original instructions', len(DAT_bigdata))})])
# concat with study data
results_df = pd.concat([results_df, pd.DataFrame({'Temperature': np.tile(None, len(dat_study)),
                                                  'Strategy': np.tile('Original instructions', len(dat_study)),
                                                  'Score': np.array(dat_study),
                                                  'Model': np.tile('Human (8k)', len(dat_study)),
                                                  'Control': np.tile('Original instructions', len(dat_study))})])

# concat with final sample
results_df = pd.concat([results_df, pd.DataFrame({'Temperature': np.tile(None, len(DAT_100k)),
                                                  'Strategy': np.tile('Original instructions', len(DAT_100k)),
                                                  'Score': np.array(DAT_100k),
                                                  'Model': np.tile('Human (100k)', len(DAT_100k)),
                                                  'Control': np.tile('Original instructions', len(DAT_100k))})])

results_df.to_csv('concatenated_results.csv', index=False)


# look for compliance

counting = {"Bard":counter_bard, "Pythia":counter_pythia, "GPT-3":counter_gpt3, "GPT-4": counter_gpt4, "Claude": counter_claude, "StableLM": counter_stablelm, "RedPajama":counter_red, "Vicuna": counter_vicuna}

# Extract data
data = {}
mean_scores = {}
for model in results_df["Model"].unique():
    # these were not used
    if "Human" in model or "StableLM-oasst" in model or 'RedPajama3B' in model:
        continue
    temp_df = results_df.loc[(results_df["Model"] == model) &
                             (results_df["Strategy"]=="Original instructions") &
                             (results_df["Temperature"]=="Mid")].dropna()
    # ratio for the given model
    data[model] = len(temp_df) / counting[model]
    mean_scores[model] = temp_df["Score"].mean(skipna=True)

# Sort data by mean score
data = dict(sorted(data.items(), key=lambda x: mean_scores[x[0]], reverse=False))


# Create a pie chart for each model
for model in counting.keys():
    # Retrieve the relevant color key
    color_key = model + "_mid_DAT"

    # Ensure the color key exists in the colors dictionary
    print(color_key)
    if color_key in colors:
        color = colors[color_key]
    else:
        color = 'lightgray'  # Default color if model color is not specified

    fig, ax = plt.subplots(figsize=(9,6))
    ax.pie([data[model], 1-data[model]], colors=['black', 'lightgray'], startangle=90)
    #plt.title("Fluent responses by model '{}' (Original instructions, Temperature: Mid)".format(model))
    plt.tight_layout()
    plt.savefig(f'{model}_piechart_bw.png', dpi=300)
    #plt.show()
    plt.close()