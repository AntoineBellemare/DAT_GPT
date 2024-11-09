import yaml
import subprocess
import itertools
import os

def run_experiment(config_file):
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    strategies = config['strategies']
    temperatures = config['temperatures']
    iterations = config['iterations']
    filename_prefix = config['filename_prefix']
    llms = config['llms']

    for llm_name, llm_config in llms.items():
        script = llm_config['script']
        for strategy, temperature, iteration in itertools.product(strategies, temperatures, range(iterations)):
            filename = f"{filename_prefix}_{llm_name}_{strategy}_temp{temperature}_{iteration}"
            command = [
                "python",
                script,
                filename,
                "--strategy", strategy,
                "--temp", str(temperature),
                "--iter_nb", str(iteration)
            ]

            print(f"Running command: {' '.join(command)}")
            try:
                subprocess.run(command, check=True, capture_output=True, text=True) # Capture output for logging
            except subprocess.CalledProcessError as e:
                print(f"Error running command: {e}")
                print(f"Stdout: {e.stdout}")
                print(f"Stderr: {e.stderr}")
                # Implement retry logic or other error handling here if needed

if __name__ == "__main__":
    run_experiment('config.yaml')
