import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from train import train_model, print_parameters, prepare_minibatch


def set_reproducibility(seed=42):
    # Seed manually to make runs reproducible
    # You need to set this again if you do multiple runs of the same model
    random.seed(seed)
    #np.random.seed(seed)
    torch.manual_seed(seed)

    # When running on the CuDNN backend two further options must be set for reproducibility
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def run_experiments(models_fns, models_args, train_data, dev_data, test_data, nums_iterations, base_name, prep_fn=prepare_minibatch, seed=42):
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    print(f'Using device: {device}')
    device = torch.device(device)

    model_names = []

    def do_train(model_fn, model_args, num_iterations):
        set_reproducibility(seed)
        model = model_fn(*model_args)
        model_names.append(model.__class__.__name__)
        print(model)
        print_parameters(model)

        model = model.to(device)

        batch_size = 25
        optimizer = optim.Adam(model.parameters(), lr=2e-4)

        # we only pass seed here for best model naming
        return train_model(model, optimizer,
                           train_data, dev_data, test_data,
                           device, seed,
                           num_iterations=num_iterations,
                           print_every=250, eval_every=250,
                           batch_size=batch_size,
                           prep_fn=prep_fn)

    results = [do_train(*funargs) for funargs in zip(models_fns, models_args, nums_iterations)]

    # save training curves
    n_models = len(models_fns)
    fig, ax = plt.subplots(n_models, 2, figsize=(2 * 6, 6 * n_models))
    if n_models == 1:  # if n_models is 1 then ax is onedimensional...
        ax[0].plot(results[0][0])
        ax[0].set_title(f'Training loss for {model_names[0]} model')
        ax[1].plot(results[0][1])
        ax[1].set_title(f'Validation accuracy for {model_names[0]} model')
    else:
        for i, (name, res) in enumerate(zip(model_names, results)):
            ax[i, 0].plot(res[0])
            ax[i, 0].set_title(f'Training loss for {name} model')
            ax[i, 1].plot(res[1])
            ax[i, 1].set_title(f'Validation accuracy for {name} model')

    plot_path = base_name + 'training_curves_seed=' + str(seed) + '.png'
    fig.savefig(plot_path)

    # save test metrics
    test_metrics = [r[2] for r in results]
    columns = ['test_acc', 'test_f1', 'test_roc_auc']
    test_metrics_df = pd.DataFrame(test_metrics, index=model_names, columns=columns)
    test_metrics_df['seed'] = seed

    test_metrics_path = base_name + 'test_metrics_seed=' + str(seed) + '.pkl'
    test_metrics_df.to_pickle(test_metrics_path)
