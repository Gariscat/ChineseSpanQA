from itertools import product
import subprocess

OPT_NAMES = ('Adam', 'SGD')
LEARNING_RATES = (1e-4, )
PRETRAINED_PATHS = ('/root/chinese-xlnet-mid', '/root/chinese-bert-wwm-ext',)

for opt_name, lr, pretrained_path in product(OPT_NAMES, LEARNING_RATES, PRETRAINED_PATHS):
    subprocess.call(f'python train.py \
                    --opt_name {opt_name} \
                    --lr {lr} \
                    --pretrained_path {pretrained_path}', \
                    shell=True
                )