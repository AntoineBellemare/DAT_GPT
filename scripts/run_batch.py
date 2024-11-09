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
            subprocess.run(command, check=True)

if __name__ == "__main__":
    run_experiment('config.yaml')
