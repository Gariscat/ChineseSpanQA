from itertools import product
import subprocess

OPT_NAMES = ('SGD',)
LEARNING_RATES = (1e-4, )
PRETRAINED_PATHS = ('/root/chinese-bert-wwm-ext',)
DROP_P = (0.25,)

for opt_name, lr, pretrained_path, drop_p in product(OPT_NAMES, LEARNING_RATES, PRETRAINED_PATHS, DROP_P):
    subprocess.call(f'python train.py \
                    --opt_name {opt_name} \
                    --lr {lr} \
                    --pretrained_path {pretrained_path} \
                    --drop_p {drop_p} \
                    --max_epochs 20', \
                    shell=True
                )
