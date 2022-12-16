import pandas as pd
from tabulate import tabulate


def metrics2df(metrics, model_names, seeds):
    metrics_rows = []
    for test_acc, f1_scores, roc_aucs in metrics:
        scores = [test_acc]
        scores.extend(f1_scores.values())
        scores.extend(roc_aucs.values())
        metrics_rows.append(scores)

    columns = ['Accuracy']
    for average in ['micro', 'macro', 'weighted']:
        columns.append(f'F1 score ({average})')
    for average in ['micro', 'macro', 'weighted']:
        columns.append(f'ROC AUC score ({average})')

    metrics_df = pd.DataFrame(metrics_rows, index=model_names, columns=columns)
    metrics_df['seed'] = seeds
    metrics_df = metrics_df.reset_index(names='Model name')
    statistics = compute_statistics(metrics_df)

    return metrics_df, statistics


def compute_statistics(metrics_df):
    grouped = metrics_df.groupby(by='Model name')
    means = grouped.mean()
    stds = grouped.std()

    fmt = '{:.4f}'
    means_str = means.applymap(lambda x: fmt.format(x))
    stds_str = stds.applymap(lambda x: fmt.format(x))
    stats = means_str + ' ± (' + stds_str + ')'
    stats = stats.drop(columns=['seed'], errors='ignore')
    return stats


def print_metrics(metrics_df):
    print(tabulate(metrics_df, headers='keys', tablefmt='psql'))
