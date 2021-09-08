"""
This is just a helper that generates scripts for submission to the HPC System.
"""
import pathlib

basepath = pathlib.Path(__file__)

basepath = str(basepath.parent).replace('scratch1', 'scratch')

tpl = """#!/bin/bash

cd {}

source bin/activate

python3.9 commit_intent_task_multi_label.py """.format(basepath)

freeze_strategy = 'no_freeze'

for model_name in ['seBERT', 'RandomForest']:
    submits = []
    for run_number in range(10):
        for fold_number in range(10):
            tmp = tpl + '{} {} {} {}'.format(model_name, run_number, fold_number, freeze_strategy)

            nodes = 'rtx5000:1'

            with open('scripts/{}_{}_{}.sh'.format(model_name, run_number, fold_number), 'w') as f:
                f.write(tmp)

            submit = 'sbatch -e {4}/logs/{0}_{1}_{2}.err -o {4}/logs/{0}_{1}_{2}.out -C scratch --mem=32G -p gpu -G {3} {4}/scripts/{0}_{1}_{2}.sh'.format(model_name, run_number, fold_number, nodes, basepath)

            # random forest does not need GPU nodes
            if model_name == 'RandomForest':
                submit = 'sbatch -e {4}/logs/{0}_{1}_{2}.err -o {4}/logs/{0}_{1}_{2}.out -C scratch --mem=32G {4}/scripts/{0}_{1}_{2}.sh'.format(model_name, run_number, fold_number, nodes, basepath)
            submits.append(submit)

    with open('commit_intent_submit_{}_{}.sh'.format(freeze_strategy, model_name), 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('\n'.join(submits))
