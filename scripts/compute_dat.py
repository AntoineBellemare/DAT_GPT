import dat
from dat_dict import chatGPT, chatGPT_etym
import numpy as np
# GloVe model from https://nlp.stanford.edu/projects/glove/
model = dat.Model("glove.840B.300d.txt", "words.txt")
print('Model loaded')
# the average DAT score for Study 1A was 78.38 (SD = 6.35)
words = chatGPT
print('Number of observations: ', len(words))
DAT_scores= []
for l in range(len(words)):
    #print(chatGPT[l][0])
    words[l] = words[l][0].split()
    DAT_scores.append(model.dat(words[l]))


print('Average DAT score: ', np.mean(DAT_scores))
