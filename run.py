from pathlib import Path
import glob
import subprocess

if __name__ == "__main__":
    config_dir = Path('./config/10000')
    configs = glob.glob(str(config_dir / '*.cfg')) 
    for config_file in configs:
        subprocess.run(['python', './ensemble.py', config_file])