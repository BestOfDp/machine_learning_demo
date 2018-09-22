from decision_tree.tree import create_tree
from decision_tree.show_tree import createPlot


def get_data_set():
    data_set = []
    labels = ['age', 'prescript', 'astigmatic', 'tearRate']
    with open('lenses.txt') as f:
        for i in f.readlines():
            data_list = i.replace('\t', ',').replace('\n', '').split(',')
            data_set.append(data_list)
    return data_set, labels


data_set, labels = get_data_set()
tree = create_tree(data_set, labels)
createPlot(tree)
