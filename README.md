# Customized OpenAI's baselines
    
## Virtual environment
Install python virtual environment
```bash
pip install virtualenv
```
Create a virtualenv with python3, one runs 
```bash
virtualenv /path/to/venv --python=python3
```
To activate a virtualenv: 
```
. /path/to/venv/bin/activate
```

## Baselines Installation
Clone the repo and cd into it:
```bash
git clone https://github.com/openai/baselines.git
cd baselines
```

If using virtualenv, create a new virtualenv and activate it
```bash
    virtualenv env --python=python3
    . env/bin/activate
```
Install baselines package
```bash
pip install -e .
```

## Run DDPG on Cutomized Environment
```bash
python -m baselines.ddpg_custom.main --env-id Ant-v2 --nb-epochs 1 --nb-epoch-cycles 1 --nb-rollout-steps 1000 --seed 0
```

## Change saving directory
In "ddpg_custom/training.py", change the following path
'/home/dlxhrl/Projects/customized_open_ai_baselines/baselines/ddpg_custom/results/ddpg_'
