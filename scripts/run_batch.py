import yaml
import subprocess
import itertools
import logging
from pathlib import Path
from typing import Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def run_command(command: list[str]) -> None:
    """Run a command with retry logic and proper logging.
    
    Args:
        command: List of command arguments to execute
    """
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logger.info(f"Command output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        logger.error(f"Stdout: {e.stdout}")
        logger.error(f"Stderr: {e.stderr}")
        raise

def build_command(script: str, filename: str, strategy: str, output_dir: str,
                 temp: float = None, model: str = None, iteration: int = 0) -> list[str]:
    """Build command list for story generation script.
    
    Args:
        script: Path to Python script to run
        filename: Output filename
        strategy: Story generation strategy
        output_dir: Output directory path
        temp: Optional temperature parameter
        model: Optional model name
        iteration: Iteration number
        
    Returns:
        List of command arguments
    """
    command = [
        "python",
        script,
        filename,
        "--strategy", strategy,
        "--file_path", output_dir,
    ]
    
    if temp is not None:
        command.extend(["--temp", str(temp)])
    if model:
        command.extend(["--model", model])
    
    command.extend(["--iter_nb", str(iteration)])
    return command

def run_experiment(config_file: str) -> None:
    """Run story generation experiments based on config file.
    
    Args:
        config_file: Path to YAML config file
    """
    config_path = Path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
        
    with open(config_path) as f:
        config = yaml.safe_load(f)

    for llm_name, llm_config in config['llms'].items():
        script = llm_config['script']
        if not Path(script).exists():
            logger.error(f"Script not found: {script}")
            continue
            
        strategies = llm_config['strategies']
        model = llm_config.get('model', llm_name)
        output_dir = Path(llm_config.get('output_dir', './machine_data_stories'))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        temperatures = llm_config.get('temperatures', [1.0])
        iterations = llm_config.get('iterations', 1)
        
        for strategy, temp, iteration in itertools.product(
            strategies, temperatures, range(iterations)):
            
            filename = f"{llm_name}_{strategy}_temp{temp}_{iteration:02d}"
            logger.info(f"Running experiment: {filename}")
            
            command = build_command(
                script=script,
                filename=filename,
                strategy=strategy,
                output_dir=str(output_dir),
                temp=temp,
                model=model,
                iteration=iteration
            )
            
            try:
                run_command(command)
                logger.info(f"Completed experiment: {filename}")
            except Exception as e:
                logger.error(f"Failed experiment {filename}: {e}")
                continue

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('batch_run.log')
        ]
    )
    run_experiment('config.yaml')
