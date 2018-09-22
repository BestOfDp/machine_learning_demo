import operator


def split_data_set(data_set, axis, value):
    """
    :param data_set: 待区分的数据集
    :param axis: 特征值
    :param value: 特征值返回的数值
    :return: 区分好的数据集
    """
    ret_data_set = []
    for data in data_set:
        if data[axis] == value:
            # 接下来两步的目的是pop data[axis]
            reduce_data = data[:axis]
            reduce_data.extend(data[axis + 1:])
            ret_data_set.append(reduce_data)
    return ret_data_set


def calc_shannon_ent(data_set):
    from math import log
    # 总数目
    sum_number = len(data_set)
    label_counts = {}

    for data in data_set:
        current_label = data[-1]
        label_counts[current_label] = label_counts.get(current_label, 0) + 1
    shanno_ent = 0.0
    for key, value in label_counts.items():
        # 求概率
        prob = float(value) / sum_number
        # 对应的公式
        shanno_ent -= prob * log(prob, 2)
    return shanno_ent


def create_data_set():
    data_set = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    labels = ['no surfacing', 'flippers']
    return data_set, labels


def choose_best_feature_to_split(data_set):
    # 除去最后一位 也就是 label位
    num_value = len(data_set[0]) - 1
    # 初始 香农熵，会和后面的进行比较
    base_entropy = calc_shannon_ent(data_set)
    best_gain = 0.0  # 当前最大的信息增益
    best_gain_index = -1  # 能够得到最大信息增益的index
    for i in range(num_value):
        # 创建唯一的分类标签列表
        data_list = [data[i] for data in data_set]
        unique_data_list = set(data_list)

        # 计算每种划分方式的信息熵
        new_entropy = 0.0
        for value in unique_data_list:
            sub_data_set = split_data_set(data_set, i, value)
            prob = len(sub_data_set) / float(len(data_set))
            new_entropy += prob * calc_shannon_ent(sub_data_set)

        # 比较，计算出最好的信息增益
        dis_gain = base_entropy - new_entropy
        if dis_gain > best_gain:
            best_gain = dis_gain
            best_gain_index = i
    return best_gain_index


def majority_cnt(classList):
    class_count = {}
    for vote in classList:
        class_count[vote] = class_count.get(vote, 0) + 1
    return sorted(class_count.items(), key=lambda x: x[1], reverse=True)[0][0]


def create_tree(data_set, label):
    # 得到一个新的labels，因为后续有del的操作，会改变原列表，会造成某些bug，所以这里创建一个新的
    labels = label[:]
    # 创建 标签，判断结束条件
    class_list = [data[-1] for data in data_set]
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]

    if len(data_set[0]) == 0:
        return majority_cnt(class_list)
    # 得到最好的划分特征索引
    best_feat = choose_best_feature_to_split(data_set)
    best_feat_label = labels[best_feat]

    my_tree = {best_feat_label: {}}
    # 与删除已经区分的label
    del labels[best_feat]

    # 获得唯一特征值
    feat_values = [data[best_feat] for data in data_set]
    feat_values_set = set(feat_values)

    for value in feat_values_set:
        # 复制标签，这样原数据不会被改变
        sub_labels = labels[:]
        my_tree[best_feat_label][value] = create_tree(
            # 根据 最好的划分数据索引 得到的特征值 来生成新的数据集
            split_data_set(data_set, best_feat, value),
            sub_labels)
    return my_tree


def classiyf(tree, labels, vecotr):
    current_node = list(tree.keys())[0]
    next_dict = tree[current_node]
    feat_index = labels.index(current_node)
    for key in next_dict.keys():
        if vecotr[feat_index] == key:
            # 当前是决策节点
            if type(next_dict[key]).__name__ == 'dict':
                class_label = classiyf(next_dict[key], labels, vecotr)
            else:
                # 当前是叶子节点的话，就返回值
                class_label = next_dict[key]
    return class_label


def store_tree(tree, filename):
    import pickle
    with open(filename, 'wb') as f:
        pickle.dump(tree, f)


def get_tree(filename):
    import pickle
    try:
        fr = open(filename)
        return pickle.load(fr)
    except Exception as e:
        return False


if __name__ == '__main__':
    data_set, label = create_data_set()
    # ent1 = calc_shannon_ent(data_set)
    # print(ent1)
    # data_set[0][-1] = 'maybe'
    # ent1 = calc_shannon_ent(data_set)
    # print(ent1)
    #
    # spilt = split_data_set(data_set, 0, 0)
    # print(spilt)
    tree = create_tree(data_set, label)
    print(classiyf(tree, label, [1, 0]))
    judge = get_tree('tree_model')
    if judge:
        tree = judge
    else:
        tree = create_tree(data_set, label)
        store_tree(tree, 'tree_model')
    # print(tree)
    # print(classiyf(tree, label, [1, 0]))
    # print(majority_cnt([1, 1, 1, 1,  2, 2, 2, 3, 2, 3, 3]))
