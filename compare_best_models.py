from train import test_eval
from evaluate import print_metrics, metrics2df
import torch
from bow import build_BOW_models
from prepare_data import load_data

device = torch.device('cpu')
train_data, _, test_data = load_data()

models_fns, models_args, _ = build_BOW_models(train_data)
# models_fns = models_fns[:1] * 3  # debug
# models_args = models_args[:1] * 3 # debug
seeds = [42, 420, 4200]
# pt_DCBOW_models, _ = build_pt_DCBOW_models()
# LSTM_models, _ = build_LSTM_models()
test_metrics = []
model_names = []
seed_vals = []

from itertools import product
for (model_fn, model_args), seed in product(zip(models_fns, models_args), seeds):
    model = model_fn(*model_args)
    model_name = model.__class__.__name__
    model_names.append(model_name)

    seed_vals.append(seed)

    file = model_name + f'_best_seed={seed}.pt'
    ckpt = torch.load(file)
    model.load_state_dict(ckpt["state_dict"])
    model_test_metrics = test_eval(model, test_data, device, batch_size=25)
    test_metrics.append(model_test_metrics)

test_metrics, statistics = metrics2df(test_metrics, model_names, seed_vals)
print_metrics(test_metrics)
print_metrics(statistics)
