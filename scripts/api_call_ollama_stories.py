import os
import time
import json
import click
import logging
import warnings
import ollama

warnings.filterwarnings('ignore')

PROMPTS = {
    "synopsis": "Write a creative and unique movie synopsis.",
    "flash-fiction": "Write a creative flash fiction story in exactly 100 words.",
    "haiku": "Write a creative haiku following the 5-7-5 syllable pattern."
}

def generate_ollama_response(prompt, temp, model_name):
    """Generates a response from Ollama."""
    try:
        response = ollama.generate(
            model=model_name,
            prompt=prompt,
            options={
                "temperature": temp if temp is not None else 1.0,
            },
        )
        return response['response'].strip()
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        time.sleep(20)  # Wait in case of rate limiting
        return None

@click.command()
@click.argument("filename", type=str)
@click.option("--file_path", type=str, help="Path to save output files")
@click.option("--strategy", type=str, default="synopsis", help="Type of story to generate")
@click.option("--temp", type=float, help="Temperature parameter")
@click.option("--iter_nb", type=str, help="Iteration number")
@click.option("--model", type=str, default="llama2", help="Ollama model to use")
def main(filename, file_path="./", strategy="synopsis", temp=None, iter_nb="0", model="llama2"):
    """
    Generate creative stories using Ollama API.
    
    Args:
        filename: Name for output file (without extension)
        file_path: Directory to save output
        strategy: Type of story (synopsis, flash-fiction, haiku)
        temp: Temperature parameter for generation
        iter_nb: Iteration number for multiple runs
        model: Ollama model name to use
    """
    logger = logging.getLogger(__name__)
    
    if strategy not in PROMPTS:
        raise ValueError(f"Invalid strategy. Must be one of {list(PROMPTS.keys())}")
    
    output = {}
    num_samples = 100  # Number of stories to generate
    
    for i in range(num_samples):
        logger.info(f"Generating story {i+1}/{num_samples}")
        
        try:
            response = generate_ollama_response(PROMPTS[strategy], temp, model)
            if response:
                output[i] = response
                logger.info(f"Successfully generated story {i+1}")
            else:
                logger.warning(f"Empty response for story {i+1}")
        except Exception as e:
            logger.error(f"Failed to generate story {i+1}: {e}")
            time.sleep(3)
            continue
        
        # Save progress after each successful generation
        output_file = os.path.join(file_path, f"{filename}.json")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, "w") as outfile:
            json.dump(output, outfile, indent=2)
        
        time.sleep(0.05)  # Rate limiting precaution
    
    logger.info(f"Generation complete. Generated {len(output)} stories.")

if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
