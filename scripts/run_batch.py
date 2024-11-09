import yaml
import subprocess
import itertools
import os

def run_experiment(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    for llm_name, llm_config in config['llms'].items():
        script = llm_config['script']
        strategies = llm_config['strategies']
        model = llm_config.get('model', llm_name)
        output_dir = llm_config.get('output_dir', './machine_data_stories')
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get optional parameters
        temperatures = llm_config.get('temperatures', [1.0])
        iterations = llm_config.get('iterations', 1)
        
        for strategy, temp, iteration in itertools.product(
            strategies, temperatures, range(iterations)):
            
            filename = f"{llm_name}_{strategy}_temp{temp}_{iteration:02d}"
            
            command = [
                "python",
                script,
                filename,
                "--strategy", strategy,
                "--file_path", output_dir,
            ]
            
            # Add optional parameters if specified
            if temp is not None:
                command.extend(["--temp", str(temp)])
            if model:
                command.extend(["--model", model])
            
            command.extend(["--iter_nb", str(iteration)])

            print(f"Running command: {' '.join(command)}")
            try:
                result = subprocess.run(command, check=True, capture_output=True, text=True)
                print(f"Output: {result.stdout}")
            except subprocess.CalledProcessError as e:
                print(f"Error running command: {e}")
                print(f"Stdout: {e.stdout}")
                print(f"Stderr: {e.stderr}")

if __name__ == "__main__":
    run_experiment('config.yaml')
