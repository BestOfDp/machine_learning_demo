import numpy as np


# 创造数据
def create_data_set():
    group = np.array([
        [1.0, 1.1],
        [1.0, 1.0],
        [0, 0],
        [0, 0.1]
    ])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(input_data, data_set, labels, k):
    data_size = data_set.shape[0]
    # 转换成与训练数据相同的shape
    diff_mat = np.tile(input_data, (data_size, 1))
    dis = diff_mat - data_set
    # 平方
    sqrt_dis = dis ** 2
    # 求和
    sum_row_dis = sqrt_dis.sum(axis=1)
    # 开方
    dis = sum_row_dis ** 0.5

    index = dis.argsort()
    cnt = {}
    for i in range(k):
        label = labels[index[i]]
        cnt[label] = cnt.get(label, 0) + 1
    # 返回频率最大的值的key
    return sorted(cnt.items(), key=lambda x: x[1], reverse=True)[0][0]


data_set, labels = create_data_set()
print("分类为:" + classify0([0, 0], data_set, labels, 3))
