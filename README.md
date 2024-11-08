# Divergent Creativity in Humans and Large Language Models

This repository contains the code and data for the paper ["Divergent Creativity in Humans and Large Language Models"](https://arxiv.org/abs/2405.13012).

## Abstract

In a computational creativity showdown, some LLMs top humans in generating diverse words but falls short in writing stories and poetry. The size of generative language models does not matter, plus humans still write more divergent haikus and synopses.

## Data

The data for this paper is located in the `human_data_dat` and `machine_data_stories` directories.

*   `human_data_dat` contains the data from the Divergent Association Task (DAT) for 100,000 human participants.
*   `human_data_synopsis` contains data for human-written synopses, scraped from IMDB
*   `machine_data_dat` contains the data from all LLMs on the DAT, >500 responses per model 
*   `machine_data_stories` contains the data from the Divergent Semantic Integration (DSI) analyses, haiku, and flash fiction tasks for a variety of LLMs.

## Code

The code for this paper is located in the `notebook` and `scripts` directories.

*   `notebook` contains Jupyter notebooks that reproduce the figures in the paper. ``dat_visualization.ipynb`` and ``dsi_visualization.ipynb`` contain main figures
*   `scripts` contains Python scripts that perform API calls to closed-source LLMs, local inference for open-source models, and the DAT (from [Olson et al. 2019](https://www.pnas.org/doi/pdf/10.1073/pnas.2022340118)) and DSI (from [Jonhson et al., 2023](https://link.springer.com/article/10.3758/s13428-022-01986-2)) computation.

**NOTE:** Some scripts are now outdated due to the fast pace of LLM development. They remain on this repo for posterity even though they can no longer be reproduced. 


## Requirements

The requirements for running the code in this repository are listed in the `requirements.txt` file.

## Running the Code

To run the code in this repository, first install the requirements:

```pip install -r requirements.txt```

and open the notebooks in your favorite editor.


