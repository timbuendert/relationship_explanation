python3.6 -m venv python_env_dygie
source python_env_dygie/bin/activate
pip install -r dygie_requirements.txt
python -m spacy download en_core_web_sm
python sample_run.py