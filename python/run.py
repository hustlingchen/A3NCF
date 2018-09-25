import subprocess

factors = [5]
activations = ['relu']
datasets = [ "Patio_Lawn_and_Garden"]
for factor in factors:
    for activation in activations:  
        for dataset in datasets:           
            subprocess.call(['python', 'ancf.py', '--dataset', dataset, '--k', str(factor), '--activation_function', activation,  '--epochs', '300', '--batch_size', '256', '--num_factors', str(factor),  '--regs', '[0,0]',  '--lr', '0.0005', '--learner', 'adam', '--verbose', '1', '--out', '1'])
