import torch.optim as optim
import matplotlib.pyplot as plt
from prepare_data import load_data
from vocabulary import build_vocabulary, build_sentiment_mappings, build_pt_embeddings
from train import train_model, print_parameters #prepare_example, simple_evaluate, get_examples
from bow import *
from lstm import LSTMClassifier
import pandas as pd

train_data, dev_data, test_data = load_data()


def run_experiments(models, nums_iterations):
    device = 'cpu'
    if torch.backends.mps.is_available():
        device = 'cpu'
    elif torch.cuda.is_available():
        device = 'cuda'

    print(f'Using device: {device}')
    device = torch.device(device)

    def do_train(model, num_iterations):
        print(model)
        print_parameters(model)

        model = model.to(device)

        batch_size = 25
        optimizer = optim.Adam(model.parameters(), lr=2e-4)

        return train_model(model, optimizer,
                           train_data, dev_data, test_data,
                           device,
                           num_iterations=num_iterations,
                           print_every=250, eval_every=250,
                           batch_size=batch_size)

    results = [do_train(model, num_iterations) for model, num_iterations in zip(models, nums_iterations)]

    n_models = len(models)
    fig, ax = plt.subplots(n_models, 2, figsize=(2 * 6, 6 * n_models))
    model_names = [m.__class__.__name__ for m in models]
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

    fig.savefig('bow_curves.png')

    # print test metrics
    test_metrics = [r[2] for r in results]
    columns = ['test_acc', 'test_f1', 'test_roc_auc']
    test_metrics_df = pd.DataFrame(test_metrics, index=model_names, columns=columns)
    print(test_metrics_df)


def build_BOW_models():
    v = build_vocabulary([train_data])
    i2t, t2i = build_sentiment_mappings()
    bow_models = [
        BOW(len(v.w2i), len(t2i), vocab=v),
        CBOW(len(v.w2i), 300, len(t2i), vocab=v),
        DeepCBOW(len(v.w2i), 300, 100, len(t2i), vocab=v),
    ]
    nums_iterations = [
        300000,
        100000,
        100000,
    ]
    return bow_models, nums_iterations


def build_PT_DCBOW():
    v_pt, vectors_pt = build_pt_embeddings()
    i2t, t2i = build_sentiment_mappings()
    return [PTDeepCBOW(len(v_pt.w2i), 300, 100, len(t2i), vocab=v_pt, vectors=vectors_pt)]


def build_LSTM():
    v_pt, vectors_pt = build_pt_embeddings()
    i2t, t2i = build_sentiment_mappings()
    return [LSTMClassifier(len(v_pt.w2i), 300, 168, len(t2i), vocab=v_pt, vectors=vectors_pt)]


if __name__ == "__main__":
    models, nums_iterations = build_BOW_models()
    # nums_iterations = [
    #     1000, 1000, 1000
    # ]
    run_experiments(models, nums_iterations)
