import numpy as np
import pandas as pd
import tensorflow as tf
import os

DATA_DIR = 'data/ratings.dat'
DATA_PATH = 'data/'
COLUMN_NAMES = ['user','item']

def re_index(s):
    i = 0
    s_map = {}
    for key in s:
        s_map[key] = i
        i += 1
    return s_map

def load_data():
    #加载数据，usecols取cols中的0,1列
    full_data = pd.read_csv(DATA_DIR,sep='::',header=None,names=COLUMN_NAMES,
                            usecols=[0,1],dtype={0:np.int32,1:np.int32},engine='python')
    full_data.user = full_data['user'] - 1
    user_set = set(full_data['user'].unique()) #统计user个数
    item_set = set(full_data['item'].unique()) #统计item个数

    user_size = len(user_set)
    item_size = len(item_set)

    item_map = re_index(item_set) ##item map一下
    item_list = []

    full_data['item'] = full_data['item'].map(lambda x:item_map[x])

    #item_set = set(full_data.item.unique())

    user_bought = {} #记录每一个用户的购买记录
    for i in range(len(full_data)):
        u = full_data['user'][i]
        t = full_data['item'][i]
        if u not in user_bought:
            user_bought[u] = []
        user_bought[u].append(t)

    user_negative = {} #记录用户在item大表中，他没有购买的东西
    for key in user_bought:
        user_negative[key] = list(item_set - set(user_bought[key]))


    user_length = full_data.groupby('user').size().tolist()
    split_train_test = []

    for i in range(len(user_set)):
        for _ in range(user_length[i] - 1):
            split_train_test.append('train')
        split_train_test.append('test')

    full_data['split'] = split_train_test
    ##从原数据集中抽取测试数据集（选取每一位用户最后一个样本作为测试数据）
    train_data = full_data[full_data['split'] == 'train'].reset_index(drop=True)
    test_data = full_data[full_data['split'] == 'test'].reset_index(drop=True)
    del train_data['split']
    del test_data['split']

    labels  = np.ones(len(train_data),dtype=np.int32)

    train_features = train_data
    train_labels = labels.tolist()

    test_features = test_data
    test_labels = test_data['item'].tolist()

    return ((train_features, train_labels),
            (test_features, test_labels),
            (user_size, item_size),
            (user_bought, user_negative))


def add_negative(features,user_negative,labels,numbers,is_training):
    feature_user,feature_item,labels_add,feature_dict = [],[],[],{}

    for i in range(len(features)):
        user = features['user'][i]
        item = features['item'][i]
        label = labels[i]

        feature_user.append(user)
        feature_item.append(item)
        labels_add.append(label)

        neg_samples = np.random.choice(user_negative[user],size=numbers,replace=False).tolist()

        if is_training:
            for k in neg_samples:
                feature_user.append(user)
                feature_item.append(k)
                labels_add.append(0)

        else:
            for k in neg_samples:
                feature_user.append(user)
                feature_item.append(k)
                labels_add.append(k)


    feature_dict['user'] = feature_user
    feature_dict['item'] = feature_item

    return feature_dict,labels_add



def dump_data(features,labels,user_negative,num_neg,is_training):
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    features,labels = add_negative(features,user_negative,labels,num_neg,is_training)

    data_dict = dict([('user', features['user']),
                      ('item', features['item']), ('label', labels)])

    print(data_dict)
    if is_training:
        np.save(os.path.join(DATA_PATH, 'train_data.npy'), data_dict)
    else:
        np.save(os.path.join(DATA_PATH, 'test_data.npy'), data_dict)



def train_input_fn(features,labels,batch_size,user_negative,num_neg):
    data_path = os.path.join(DATA_PATH, 'train_data.npy')
    if not os.path.exists(data_path):
        dump_data(features, labels, user_negative, num_neg, True)

    data = np.load(data_path).item()

    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.shuffle(100000).batch(batch_size)

    return dataset


def eval_input_fn(features, labels, user_negative, test_neg):
    """ Construct testing dataset. """
    data_path = os.path.join(DATA_PATH, 'test_data.npy')
    if not os.path.exists(data_path):
        dump_data(features, labels, user_negative, test_neg, False)

    data = np.load(data_path).item()
    print("Loading testing data finished!")
    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.batch(test_neg + 1)

    return dataset