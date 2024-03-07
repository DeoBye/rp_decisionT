from pathlib import Path
import glob
import subprocess

if __name__ == "__main__":
    config_dir = Path('./config')
    configs = glob.glob(str(config_dir / '*.cfg'))
    print(configs) 
    for config_file in configs:
        print(config_file)
        subprocess.run(['python', './ensemble.py', config_file])