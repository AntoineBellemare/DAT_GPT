import json
import glob
import fnmatch
import pandas as pd
import time
import numpy as np
from antropy import lziv_complexity
import torch
from transformers import BertModel, BertTokenizer
from nltk.tokenize import PunktSentenceTokenizer
import string
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="numba")
warnings.filterwarnings("ignore")

cos = torch.nn.CosineSimilarity(dim = 0)

def initialize_model():
    # initialize BERT model instance
    model = BertModel.from_pretrained("bert-large-uncased", output_hidden_states=True)
    model.eval()
    return model

def initialize_tokenizer():
    # initialize BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    return tokenizer

def load_data(filename):
    # load data
    if "TMDB" in filename:
        data = pd.read_json(filename)
    else:
        with open(filename) as json_file:
            data = json.load(json_file)
    return data

def train_segmenter(segmenter, text):
    # train the segmenter on the text first (contextual embedding)
    segmenter.train(text)

def segment_text(segmenter, text):
    # tokenize text into sentences
    sentences = segmenter.tokenize(text)
    return sentences

def get_sentence_features(sentences, tokenizer, model):
    """
    Get BERT features for each sentence in a story.
    
    Parameters
    ----------
    sentences : list
        List of sentences in a story
    tokenizer : BertTokenizer
        BERT tokenizer
    model : BertModel
        BERT model
    
    Returns
    -------
    features : list
        List of BERT features for each sentence
    words : list
        List of words in story"""
    features = []
    words = []
    filter_list = np.array(['[CLS]', '[PAD]', '[SEP]', '.', ',', '!', '?'])
    for i in range(len(sentences)):
        sentence = sentences[i].translate(str.maketrans('', '', string.punctuation))
        sent_tokens = tokenizer(sentence, max_length=50, truncation=True, padding='max_length', return_tensors="pt")
        sent_words = [tokenizer.decode([k]) for k in sent_tokens['input_ids'][0]]
        sent_indices = np.where(np.in1d(sent_words, filter_list, invert=True))[0]

        with torch.no_grad():
            sent_output = model(**sent_tokens)
            hids = sent_output.hidden_states
        layer6 = hids[6]
        layer7 = hids[7]

        for j in sent_indices:
            words.append(sent_words[j])
            words.append(sent_words[j])
            features.append(layer6[0, j, :])
            features.append(layer7[0, j, :])

    return features, words
def calculate_lziv(text):
    """
    Calculate the LZIV complexity of a story.

    Parameters
    ----------
    text : str
        Story text. Raw text or tokenized text.
    """
    lziv = lziv_complexity(text, normalize=True)
    return lziv

def calculate_dcos(features, words):
    """
    Calculate the DSI score of a story.
    
    Parameters
    ----------
    features : list
        List of sentence embeddings
    words : list
        List of words
    
    Returns
    -------
    mean_story_dcos : float
        DSI score of story
    Reference
    ---------
    Johnson et al., 2022 Extracting Creativity from Narratives using Distributional Semantic Modeling
    https://osf.io/ath2s/
    """
    num_words = len(words)
    sentence_embeddings = [feature.numpy() for feature in features]
    
    # calculate DSI
    lower_triangle_indices = np.tril_indices_from(np.random.rand(num_words, num_words), k=-1)
    story_dcos_vals = []
    for k in range(len(lower_triangle_indices[0])):
        features1 = features[lower_triangle_indices[0][k]]
        features2 = features[lower_triangle_indices[1][k]]
        dcos = (1 - cos(features1, features2))
        story_dcos_vals.append(dcos)
    mean_story_dcos = torch.mean(torch.stack(story_dcos_vals)).item()
    return mean_story_dcos, num_words

def apply_clustering(features, sentences):
    """
    Apply clustering to the sentence embeddings.
    
    Parameters
    ----------
    features : list
        List of sentence embeddings
    sentences : list
        List of sentences
    
    Returns
    -------
    embeddings_2d : np.array
        2D embeddings of sentences
    cluster_labels : np.array
        Cluster labels of sentences
    """
    sentence_embeddings = [feature.numpy() for feature in features]
    k = len(sentences)
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(sentence_embeddings)
    cluster_labels = kmeans.labels_

    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(sentence_embeddings)

    return embeddings_2d, cluster_labels

def plot_cluster_plot(embeddings_2d, cluster_labels, dsi, sentences, output_path):
    """
    Plot the cluster plot.
    
    Parameters
    ----------
    embeddings_2d : np.array
        2D embeddings of sentences
    cluster_labels : np.array
        Cluster labels of sentences
    dsi : float
        DSI score of story
    sentences : list
        List of sentences
    output_path : str
        Path to save plot
    """
    fig, ax = plt.subplots()
    color_map = plt.cm.get_cmap('viridis', len(sentences))
    colors = color_map(cluster_labels)
    scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors)

    for i in range(len(embeddings_2d) - 1):
        x1, y1 = embeddings_2d[i]
        x2, y2 = embeddings_2d[i + 1]
        ax.plot([x1, x2], [y1, y2], 'b-', alpha=0.2)

    ax.set_xlim([np.min(embeddings_2d[:, 0]), np.max(embeddings_2d[:, 0])])
    ax.set_ylim([np.min(embeddings_2d[:, 1]), np.max(embeddings_2d[:, 1])])
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title(f"Cluster Plot: reduced embeddings \nScore: {dsi}\nNumber of sentences: {len(sentences)}")

    cbar = plt.colorbar(scatter)
    cbar.set_label('Sentence Cluster')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def process_files(filenames):
    """
    Process files and return a dictionary of story IDs and their DSI scores.

    DSI scores are calculated using the following steps:
    1. Tokenize sentences using NLTK's PunktSentenceTokenizer
    2. Extract BERT features for each sentence
    3. Calculate DSI score of story based on the cosine similarity between sucessive words embeddings
    4. Apply clustering to the sentence embeddings
    5. Plot the cluster plot
    6. Save the data to a csv file

    Parameters
    ----------
    filenames : list
        List of filenames to process
    """
    model = initialize_model()
    tokenizer = initialize_tokenizer()
    segmenter = PunktSentenceTokenizer()
    s = {}
    counter = 0
    for filename in filenames:
        print(f"Processing file: {filename}")
        # get info from filename
        # get model name,
        if 'GPT4' in filename:
            model_name = 'GPT4'
        elif 'GPT3' in filename:
            model_name = 'GPT3'
        elif 'Vicuna' in filename:
            model_name = 'Vicuna'
        elif 'human' in filename:
            model_name = 'human'
        # get creative writing condition
        if 'synopses' in filename:
            condition = 'synopsis'
        elif 'flash_fictions' in filename:
            condition = 'flash-fiction'
        elif 'haikus' in filename:
            condition = 'haiku'
        #else:
        #    condition = 'synopsis'
        # get temperature
        if 'temp1.0' in filename:
            temp = 'Mid'
        elif 'temp1.2' in filename and 'flash' in filename:
            temp = 'Very High'
        elif 'temp1.2' in filename:
            temp = 'High'
        elif 'temp0.8' in filename and 'flash' in filename:
            temp = 'Very Low'
        elif 'temp0.8' in filename:
            if model_name == 'Vicuna':
                temp = 'Mid'
            else:
                temp = 'Low'
        elif 'temp0.6' in filename:
            temp = 'Very Low'
        elif 'temp1.5' in filename or 'temp1.4' in filename:
            temp = 'Very High'
        elif 'temp0.9' in filename:
            temp = 'Low'
        elif 'temp1.1' in filename:
            temp = 'High'
        else:
            temp = 'n.a.'
        
        # load data
        try:
            data = load_data(filename)
        except IndexError:
            print(f"Error loading data; no {filename}")
            continue
        if model_name == 'IMDB' or 'haikus' in filename:
            iterator = range(0, len(data)-1)
        else:
            print(model_name)
            iterator = data.keys()
        for index in iterator:
            last_time = time.time()
            ID = counter
            try:
                text = data[index]
            except KeyError:
                text = data.iloc[index]["overview"]
            if text == "":
                continue
            s[ID] = {}
            print(f"Processing story for story {str(ID)}\n"
                  f"Story: {text}")

            # get sentence features
            train_segmenter(segmenter, text)
            sentences = segment_text(segmenter, text)
            lziv = calculate_lziv(text)
            features, words = get_sentence_features(sentences, tokenizer, model)
            
            # calculate DSI
            mean_story_dcos, num_words = calculate_dcos(features, words)
            s[ID]["DSI"] = mean_story_dcos
            print(f"DSI for participant {str(ID)}: {mean_story_dcos}")
            print(f"Number of words: {num_words}")
            print(f"Lempel-Ziv: {lziv}")
            
            # log data
            s[ID]["story"] = text
            s[ID]["model"] = model_name
            s[ID]["condition"] = condition
            s[ID]["temp"] = temp
            s[ID]["num_words"] = num_words
            s[ID]["lziv"] = lziv

            #embeddings_2d, cluster_labels = apply_clustering(features, sentences)
            #output_path = f"./figures/{condition}/{model_name}_sample{str(ID)}_temp-{temp}_clusters.png"
            #plot_cluster_plot(embeddings_2d, cluster_labels, mean_story_dcos, sentences, output_path)
            counter += 1
            elapsed_time = time.time() - last_time
            print('Elapsed time for story: ' + str(elapsed_time))
            elapsed_time = time.time() - start_time
            print('Elapsed time since beginning: ' + str(elapsed_time))


            dsi_df = pd.DataFrame.from_dict(s, orient="index")
            dsi_df.to_csv('machine_DSI-lziv_output.csv', index=False)
        elapsed_time = time.time() - start_time
        print('Elapsed time: ' + str(elapsed_time))


# USER EDIT
filenames = glob.glob("../machine_data_stories/final/*.json")
#filenames = glob.glob("./human_data_synopsis/TMDB_movies_subset.json")
print(f"Number of files to process: {len(filenames)}")

start_time = time.time()
process_files(filenames)
