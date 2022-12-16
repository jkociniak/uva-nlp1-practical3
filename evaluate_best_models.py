from train import test_eval
import torch
from run_experiments import build_BOW_models, build_pt_DCBOW_models, build_LSTM_models
from prepare_data import load_data
import pandas as pd

device = torch.device('cpu')
_, _, test_data = load_data()

BOW_models, _ = build_BOW_models()
pt_DCBOW_models, _ = build_pt_DCBOW_models()
LSTM_models, _ = build_LSTM_models()

models = BOW_models + pt_DCBOW_models + LSTM_models
test_metrics = []
model_names = []
for model in models:
    model_name = model.__class__.__name__
    model_names.append(model_name)
    file = 'saved/' + model_name + '.pt'
    ckpt = torch.load(file)
    model.load_state_dict(ckpt["state_dict"])
    test_metrics.append(test_eval(model, test_data, device, batch_size=25))

columns = ['test_acc', 'test_f1', 'test_roc_auc']
test_metrics_df = pd.DataFrame(test_metrics, index=model_names, columns=columns)
print(test_metrics_df)