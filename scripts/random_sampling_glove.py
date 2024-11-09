import random
import pandas as pd
from dat import Model  # Assuming the Model class is saved in a file named dat_model.py

# Load words from the words.txt file
with open('./model/words.txt', 'r') as file:
    words = file.read().splitlines()

# Initialize the model (ensure the correct path to the GloVe model and words.txt file)
model = Model(model="./model/glove.840B.300d.txt", dictionary='./model/words.txt')
print("Model loaded successfully!")

# Function to randomly sample 10 words and compute DAT score
def sample_and_compute_dat(model, words, num_samples=5000, sample_size=10):
    samples = []
    scores = []
    for _ in range(num_samples):
        sample = random.sample(words, sample_size)
        score = model.dat(sample)
        print(f"Sample: {sample}, Score: {score}")
        samples.append(sample)
        scores.append(score)
    return samples, scores

# Sample words and compute DAT scores
samples, scores = sample_and_compute_dat(model, words)

# Prepare data for DataFrame
data = {
    'Temperature': ['Mid'] * len(samples),
    'Strategy': ['Original Instructions'] * len(samples),
    'Score': scores,
    'Model': ['Randomization'] * len(samples),
    'Control': ['Randomization'] * len(samples),
    'Words': [sample for sample in samples]
}

# Create DataFrame
df = pd.DataFrame(data)

# Display the first few rows of the DataFrame
print(df.head())

# Save DataFrame to CSV
df.to_csv('random_glove_dat_scores.csv', index=False)
