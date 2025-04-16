# Leveraging LLMs and attention-mechanism for automatic annotation of historical maps

# Installation
```shell
conda create -n llm_attn python=3.9
conda activate llm_attn
pip install -r requirements.txt
```

# Dataset
Download the [Hameln dataset](https://seafile.cloud.uni-hannover.de/f/af2a925049a047a9a97e/?dl=1).

# Train
Examples of config files are in the `config` directory.

```shell
python train.py --config [CONFIG_FILE]
```