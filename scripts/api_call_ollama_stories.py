import os
import time
import json
import click
import logging
import warnings
from pathlib import Path
from typing import Dict, Optional
from tenacity import retry, stop_after_attempt, wait_exponential
import ollama

warnings.filterwarnings('ignore')

PROMPTS = {
    "synopsis": "Write a creative and unique movie synopsis.",
    "flash-fiction": "Write a creative flash fiction story in exactly 100 words.",
    "haiku": "Write a creative haiku following the 5-7-5 syllable pattern."
}

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_ollama_response(prompt: str, temp: Optional[float], model_name: str) -> Optional[str]:
    """
    Generate a response from Ollama with retry logic.
    
    Args:
        prompt: The prompt to send to the model
        temp: Temperature parameter for generation
        model_name: Name of the Ollama model to use
        
    Returns:
        Generated text response or None if failed
    """
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
        logging.error(f"Error generating response: {str(e)}")
        raise

def save_progress(output: Dict, filepath: Path) -> None:
    """Save current progress to JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(output, f, indent=2)

@click.command()
@click.argument("filename", type=str)
@click.option("--file_path", type=str, default="./", help="Path to save output files")
@click.option("--strategy", type=str, default="synopsis", help="Type of story to generate")
@click.option("--temp", type=float, help="Temperature parameter")
@click.option("--iter_nb", type=str, default="0", help="Iteration number")
@click.option("--model", type=str, default="llama2", help="Ollama model to use")
@click.option("--num_samples", type=int, default=100, help="Number of samples to generate")
def main(filename: str, file_path: str="./", strategy: str="synopsis", 
         temp: Optional[float]=None, iter_nb: str="0", model: str="llama2",
         num_samples: int=100) -> None:
    """
    Generate creative stories using Ollama API.
    
    Args:
        filename: Name for output file (without extension)
        file_path: Directory to save output
        strategy: Type of story (synopsis, flash-fiction, haiku)
        temp: Temperature parameter for generation
        iter_nb: Iteration number for multiple runs
        model: Ollama model name to use
        num_samples: Number of samples to generate
    """
    logger = logging.getLogger(__name__)
    
    if strategy not in PROMPTS:
        raise ValueError(f"Invalid strategy. Must be one of {list(PROMPTS.keys())}")
    
    output_path = Path(file_path) / f"{filename}.json"
    output: Dict = {}
    
    # Resume from existing progress if file exists
    if output_path.exists():
        with open(output_path) as f:
            output = json.load(f)
        start_idx = max(map(int, output.keys())) + 1 if output else 0
        logger.info(f"Resuming from sample {start_idx}")
    else:
        start_idx = 0
    
    for i in range(start_idx, num_samples):
        logger.info(f"Generating story {i+1}/{num_samples}")
        
        try:
            response = generate_ollama_response(PROMPTS[strategy], temp, model)
            if response:
                output[str(i)] = response
                logger.info(f"Successfully generated story {i+1}")
                # Save progress after each successful generation
                save_progress(output, output_path)
            else:
                logger.warning(f"Empty response for story {i+1}")
        except Exception as e:
            logger.error(f"Failed to generate story {i+1}: {str(e)}")
            continue
        
        time.sleep(0.05)  # Rate limiting precaution
    
    logger.info(f"Generation complete. Generated {len(output)} stories.")

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('ollama_stories.log')
        ]
    )
    main()
