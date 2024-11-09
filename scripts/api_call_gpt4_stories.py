import json
import click
import time
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.getenv('OPENAI_API_KEY'),
)

PROMPTS = {
    "synopsis": "Write a creative and unique movie synopsis.",
    "flash-fiction": "Write a creative flash fiction story in exactly 100 words.",
    "haiku": "Write a creative haiku following the 5-7-5 syllable pattern."
}

def generate_response(prompt, temp, model_name):
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temp,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating response: {e}")
        time.sleep(20)
        return None

@click.command()
@click.argument('filename')
@click.option('--file_path', default="./", help='Path to save the output file')
@click.option('--strategy', default='synopsis', help='Type of story to generate')
@click.option('--temp', default=None, type=float, help='Temperature parameter')
@click.option('--iter_nb', default='0', help='Iteration number')
@click.option('--model', default='gpt-4', help='Model name to use')
def main(filename, file_path="./", strategy='synopsis', temp=None, iter_nb='0', model='gpt-4'):
    if strategy not in PROMPTS:
        raise ValueError(f"Invalid strategy. Must be one of {list(PROMPTS.keys())}")
    
    output_file = f"{file_path}/machine_data_stories/{filename}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    responses = {}
    num_samples = 100
    
    for i in range(num_samples):
        print(f"Generating story {i+1}/{num_samples}")
        response = generate_response(PROMPTS[strategy], temp, model)
        if response:
            responses[i] = response
        time.sleep(1)
        
    with open(output_file, 'w') as f:
        json.dump(responses, f, indent=2)

if __name__ == '__main__':
    main()
