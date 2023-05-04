from llama_cpp import Llama

llm = Llama(model_path="ggml-alpaca-7b-q4.bin")

prompt = "Please enter 10 words that are as different from each other as possible, in all meanings and uses of the words. Rules: Only single words in English. Only nouns (e.g., things, objects, concepts). No proper nouns (e.g., no specific people or places). No specialised vocabulary (e.g., no technical terms). Think of the words on your own (e.g., do not just look at objects in your surroundings).  Make a list of these 10 words, a single word in each entry of the list. Output these words as a comma separated list."

result_file = open("results-7b.txt", "a")
for i in range(500):
  output = llm(f"### Instruction:\n\n{prompt}\n\n### Response:\n\n", max_tokens=128, stop=["###"])
  result_file.write(output["choices"][0]["text"].strip() + "\n")
  result_file.flush()
