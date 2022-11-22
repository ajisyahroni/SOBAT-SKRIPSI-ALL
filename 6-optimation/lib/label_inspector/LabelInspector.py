from rich import print as rprint
from rich.console import Console
from rich import inspect
import seaborn as sns
from matplotlib import pyplot as plt
console = Console()


def _myprint(title, object):
    rprint(f'[bold magenta]{title}[/bold magenta]', object)


def _get_redundant_pairs(dataset):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = dataset.columns
    for i in range(0, dataset.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop


def _get_top_abs_correlations(dataset, n=5):
    au_corr = dataset.corr().unstack()
    labels_to_drop = _get_redundant_pairs(dataset)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]


def inspect(dataset, labels, column_name, show_img=False):
    # get unlabel doc
    dataset_len = len(dataset)
    condition_str = ''
    for i in range(0, len(labels)):
        condition_str += f'(dataset["{labels[i]}"] != 1)'
        if(i < len(labels)-1):
            condition_str += '&'
    unlabel_docs = dataset[eval(condition_str)]
    unlabel_percentage = round(len(unlabel_docs)/dataset_len * 100, 3)

    # jumlah dataset yang atribut utamanya null
    null_docs = dataset[dataset[column_name].isnull()]

    _myprint('% UNLABEL DOCS:', unlabel_percentage)
    _myprint(f'NULL {column_name} DOCS:', len(null_docs))

    # total label
    _myprint(f'NUMBER OF DOC/LABEL : \n', dataset[labels].sum())
    _myprint(f'MULTI-LABEL COMBINATION : \n', dataset[labels].value_counts())

    # get best correlation
    _myprint('5 BEST LABEL CORRELATION\n',_get_top_abs_correlations(dataset[labels], 5))

    if(show_img == True):
        # 5. check correlation of labels
        data = dataset[labels]
        cmap = plt.cm.plasma
        plt.figure(figsize=(7, 7))
        plt.title('korelasi dari fitur dan label')
        sns.heatmap(data.astype(float).corr(), linewidths=.1,
                    vmax=1.0, square=True, cmap=cmap, linecolor='white', annot=True)
        plt.show()

        # 4. check character length
        dataset['char_length'] = dataset[column_name].apply(
            lambda x: len(str(x)))
        sns.set()
        dataset['char_length'].hist()
        plt.show()
