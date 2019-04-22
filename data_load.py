import json
import os
import pickle

def get_data_path(data_name, src_site):
    train_datadir = './datasets/rumoureval-2019-training-data'
    test_datadir = './datasets/rumoureval-2019-test-data'

    if data_name == 'train' and src_site == 'twitter':
        path = os.path.join(train_datadir, 'twitter-english')
        keys, _ = get_train_src_keys(train_datadir, split=True)
    elif data_name == 'train' and src_site == 'reddit':
        path = os.path.join(train_datadir, 'reddit-training-data')
        _, keys = get_train_src_keys(train_datadir, split=True)
    elif data_name == 'dev' and src_site == 'twitter':
        path = os.path.join(train_datadir, 'twitter-english')
        keys, _ = get_dev_src_keys(train_datadir, split=True)
    elif data_name == 'dev' and src_site == 'reddit':
        path = os.path.join(train_datadir, 'reddit-dev-data')
        _, keys = get_dev_src_keys(train_datadir, split=True)
    elif data_name == 'test' and src_site == 'twitter':
        path = os.path.join(test_datadir, 'twitter-en-test-data')
        keys = None
    elif data_name == 'test' and src_site == 'reddit':
        path = os.path.join(test_datadir, 'reddit-test-data')
        keys = None
    else:
        LookupError
    return path, keys

def get_categories(data_name, src_site):
    data_path, _ = get_data_path(data_name, src_site)
    categories = os.listdir(data_path)
    return categories

def get_src_dirs(data_name, src_site):
    categories = get_categories(data_name, src_site)
    src_dirs = []   
    if data_name != 'test':
        data_path, keys = get_data_path(data_name, src_site)
        for cate in categories:
            cate_path = os.path.join(data_path, cate)
            if src_site == 'twitter':
                src_dirs += [os.path.join(cate_path, src_id) for src_id in os.listdir(cate_path) if src_id in keys]
            elif src_site == 'reddit':
                src_dirs.append(cate_path)
            else:
                LookupError
    else:
        data_path, _ = get_data_path(data_name, src_site)
        for cate in categories:
            cate_path = os.path.join(data_path, cate)
            if src_site == 'twitter':
                src_dirs += [os.path.join(cate_path, src_id) for src_id in os.listdir(cate_path)]
            elif src_site == 'reddit':
                src_dirs.append(cate_path)
    return src_dirs

def get_train_src_keys(train_datadir, split=False):
    with open(os.path.join(train_datadir, 'train-key.json'), 'r') as f:
        keys = json.load(f)
        ids = keys['subtaskbenglish'].keys()
        if split:
            twit_ids, redit_ids = twitter_reddit_split(list(ids))
            return twit_ids, redit_ids
        else:
            return list(ids)

def get_dev_src_keys(train_datadir, split=False):
    with open(os.path.join(train_datadir, 'dev-key.json'), 'r') as f:
        keys = json.load(f)
        ids = keys['subtaskbenglish'].keys()
        if split:
            twit_ids, redit_ids = twitter_reddit_split(list(ids))
            return twit_ids, redit_ids
        else:
            return list(ids)

def twitter_reddit_split(ids):
    twit_ids = []
    redit_ids = []
    for id in ids:
        if len(id) == 18:
            twit_ids.append(id)
        elif len(id) > 1:
            redit_ids.append(id)
        else:
            return ValueError
    return twit_ids, redit_ids

def dict_save_to_pkl(saved_path, dict):
    with open(saved_path, 'wb') as f:
        pickle.dump(dict, f)

def get_all_replies_dict(data_name, src_sites):
    if 'pkl_saved' in os.listdir('./datasets'):
        if './datasets/pkl_saved/%s-%s-replies-paths.pkl'%(data_name, src_sites) in os.listdir('./datasets/pkl_saved'):
            saved_path = './datasets/pkl_saved/%s-%s-replies-paths.pkl'%(data_name, src_sites)
            with open(saved_path, 'rb') as f:
                all_replies_path = pickle.load(f)
            return all_replies_path
        else:
            src_dirs = get_src_dirs(data_name, src_sites)
            all_replies_paths = {}
            for dir in src_dirs:
                file_names = os.listdir(os.path.join(dir, 'replies'))
                for file_name in file_names:
                    id = file_name.split('.json')[0]
                    all_replies_paths[id] = os.path.join(dir, 'replies', file_name)
            dict_save_to_pkl('./datasets/pkl_saved/%s-%s-replies-paths.pkl'%(data_name, src_sites), all_replies_paths)
            return all_replies_paths
    else:
        os.mkdir('./datasets/pkl_saved')
        return get_all_replies_dict(data_name, src_sites)


def get_all_sources_dict(data_name, src_sites):
    if 'pkl_saved' in os.listdir('./datasets'):
        if './datasets/pkl_saved/%s-%s-sources-paths.pkl'%(data_name, src_sites) in os.listdir('./datasets/pkl_saved'):
            saved_path = './datasets/pkl_saved/%s-%s-sources-paths.pkl'%(data_name, src_sites)
            with open(saved_path, 'rb') as f:
                all_sources_path = pickle.load(f)
            return all_sources_path
        else:
            src_dirs = get_src_dirs(data_name, src_sites)
            all_sources_paths = {}
            for dir in src_dirs:
                file_name = os.listdir(os.path.join(dir, 'source-tweet'))
                id = file_name[0].split('.json')[0]
                all_sources_paths[id] = os.path.join(dir, 'source-tweet', file_name[0])
            dict_save_to_pkl('./datasets/pkl_saved/%s-%s-sources-paths.pkl'%(data_name, src_sites), all_sources_paths)
            return all_sources_paths
    else:
        os.mkdir('./datasets/pkl_saved')
        return get_all_sources_dict(data_name, src_sites)


def get_all_sources_ids(data_name, src_sites):
    all_sources_dict = get_all_sources_dict(data_name, src_sites)
    return list(all_sources_dict.keys())

def get_all_sources_paths(data_name, src_sites):
    all_sources_dict = get_all_sources_dict(data_name, src_sites)
    return list(all_sources_dict.values())

def get_all_replies_ids(data_name, src_sites):
    all_replies_dict = get_all_replies_dict(data_name, src_sites)
    return list(all_replies_dict.keys())

def get_all_replies_paths(data_name, src_sites):
    all_replies_dict = get_all_replies_dict(data_name, src_sites)
    return list(all_replies_dict.values())

def get_one_source_path(data_name, src_sites, id):
    all_sources_dict = get_all_sources_dict(data_name, src_sites)
    return all_sources_dict.get(id)

def get_one_reply_path(data_name, src_sites, id):
    all_replies_dict = get_all_replies_dict(data_name, src_sites)
    return all_replies_dict.get(id)

def get_one_source_child_ids(data_name, src_sites, id):
    src_path = get_one_source_path(data_name, src_sites, id)
    dir = src_path.split('/source-tweet')[0]
    file_names = os.listdir(os.path.join(dir, 'replies'))
    return [file_name.split('.json')[0] for file_name in file_names]

def get_one_reply_parent_id(data_name, src_sites, id):
    reply_path = get_one_reply_path(data_name, src_sites, id)
    dir = reply_path.split('/replies')[0]
    parent_id = dir.split('/')[-1]
    return parent_id

def get_one_source_category(data_name, src_sites, id):
    src_path = get_one_source_path(data_name, src_sites, id)
    category = src_path.split('/')[-4]
    return category

def get_one_reply_category(data_name, src_sites, id):
    reply_path = get_one_reply_path(data_name, src_sites, id)
    category = reply_path.split('/')[-4]
    return category
