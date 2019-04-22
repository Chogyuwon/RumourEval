import json
import os
import pickle
import math

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
            for src_dir in src_dirs:
                src_id = src_dir.split('/')[-1]
                all_replies_paths[src_id] = os.path.join(src_dir, 'source-tweet', src_id+'.json')
                file_names = os.listdir(os.path.join(src_dir, 'replies'))
                for file_name in file_names:
                    id = file_name.split('.json')[0]
                    all_replies_paths[id] = os.path.join(src_dir, 'replies', file_name)
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
    parent_id = parent_id.replace('.json', '')
    return parent_id

def get_one_source_category(data_name, src_sites, id):
    src_path = get_one_source_path(data_name, src_sites, id)
    category = src_path.split('/')[-4]
    return category

def get_one_reply_category(data_name, src_sites, id):
    reply_path = get_one_reply_path(data_name, src_sites, id)
    category = reply_path.split('/')[-4]
    return category

def get_one_reply_label(data_name, id):
    train_datadir = './datasets/rumoureval-2019-training-data'
    if data_name == "train":
        with open(os.path.join(train_datadir, 'train-key.json'), 'r') as f:
            keys = json.load(f)
        return keys['subtaskaenglish'].get(id)
    elif data_name == "dev":
        with open(os.path.join(train_datadir, 'dev-key.json'), 'r') as f:
            keys = json.load(f)
        return keys['subtaskaenglish'].get(id)
    else:
        ValueError

def get_one_source_label(data_name, id):
    train_datadir = './datasets/rumoureval-2019-training-data'
    if data_name == "train":
        with open(os.path.join(train_datadir, 'train-key.json'), 'r') as f:
            keys = json.load(f)
        return keys['subtaskbenglish'].get(id)
    elif data_name == "dev":
        with open(os.path.join(train_datadir, 'dev-key.json'), 'r') as f:
            keys = json.load(f)
        return keys['subtaskbenglish'].get(id)
    else:
        ValueError

def parsing_tweet(data_name, src_sites, id, src):
    if src:
        with open(get_one_source_path(data_name, src_sites, id), 'r') as f:
            tweet = json.load(f)
        label = get_one_source_label(data_name, id)
        if src_sites == 'twitter':
            id = tweet['id']
            src_text = tweet['text'].replace('\n', '')
        elif src_sites == 'reddit':
            id = tweet['data']['children'][0]['data']['id']
            # id = int.from_bytes(tweet['data']['children'][0]['data']['id'].encode(), 'little')
            src_text = tweet['data']['children'][0]['data']['title'].replace('\n', '')
            # id_decode = id.to_bytes(math.ceil(id.bit_length() / 8), 'little').decode()
        else:
            ValueError
        result = {"id": id, "data_name":data_name, "src_sites":src_sites, "label":label, "text_a":src_text}
    else:           
        with open(get_one_reply_path(data_name, src_sites, id), 'r') as f:
            tweet = json.load(f)
        label = get_one_reply_label(data_name, id)
        if src_sites == 'twitter':
            id_str = str(tweet['id'])
            src_tweet_id = get_one_reply_parent_id(data_name, src_sites, id_str)
            src_tweet_path = get_one_source_path(data_name, src_sites, src_tweet_id)
            with open(src_tweet_path, 'r') as f:
                src_tweet = json.load(f)
            id = id_str
            src_text = src_tweet['text'].replace('\n', '')
            try:
                reply_text = tweet['text'].replace('\n', '')
            except:
                reply_text = ' '

        elif src_sites =='reddit':
            try:
                id_str = tweet['data']['id']
            except:
                id_str = tweet['data']['children'][0]['data']['id']
            # id = int.from_bytes(tweet['data']['id'].encode(), 'little')
            id = id_str
            src_tweet_id = get_one_reply_parent_id(data_name, src_sites, id_str)
            src_tweet_path = get_one_source_path(data_name, src_sites, src_tweet_id)
            with open(src_tweet_path, 'r') as f:
                src_tweet = json.load(f)

            src_text = src_tweet['data']['children'][0]['data']['title'].replace('\n', '')
            try:
                reply_text = tweet['data']['body'].replace('\n', '')
            except:
                reply_text = ' '
        result = {"id": id, "data_name":data_name, "src_sites":src_sites, "label":label, "text_a":src_text, 'text_b':reply_text}

    return result

class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

def make_examples(data_name):
    twit_replies_ids = get_all_replies_ids(data_name, 'twitter')
    redit_replies_ids = get_all_replies_ids(data_name, 'reddit')

    examples = []
    for id in twit_replies_ids:
        parsed_tweet = parsing_tweet(data_name, 'twitter', id, src=False)
        guid = parsed_tweet['id']
        text_a = parsed_tweet['text_a']
        text_b = parsed_tweet['text_b']
        if parsed_tweet['label'] == "support":
            label = '0'
        elif parsed_tweet['label'] == "deny":
            label = '1'
        elif parsed_tweet['label'] == 'query':
            label = '2'
        elif parsed_tweet['label'] == 'comment':
            label = '3'
        else:
            ValueError
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
        )


    for id in redit_replies_ids:
        parsed_tweet = parsing_tweet(data_name, 'reddit', id, src=False)
        guid = int.from_bytes(parsed_tweet['id'].encode(), 'little')           
        text_a = parsed_tweet['text_a']
        text_b = parsed_tweet['text_b']
        if parsed_tweet['label'] == "support":
            label = '0'
        elif parsed_tweet['label'] == "deny":
            label = '1'
        elif parsed_tweet['label'] == 'query':
            label = '2'
        elif parsed_tweet['label'] == 'comment':
            label = '3'
        else:
            ValueError
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
        )
    return examples

def make_test_examples():
    twit_replies_ids = get_all_replies_ids('test', 'twitter')
    redit_replies_ids = get_all_replies_ids('test', 'reddit')

    examples = []
    for id in twit_replies_ids:
        parsed_tweet = parsing_tweet('test', 'twitter', id, src=False)
        guid = parsed_tweet['id']
        text_a = parsed_tweet['text_a']
        text_b = parsed_tweet['text_b']
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=text_b, label=None)
        )


    for id in redit_replies_ids:
        parsed_tweet = parsing_tweet('test', 'reddit', id, src=False)
        guid = parsed_tweet['id']
        text_a = parsed_tweet['text_a']
        text_b = parsed_tweet['text_b']
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=text_b, label=None)
        )
    return examples


# if __name__ == "__main__":
    # train_examples = make_examples('train')
    # print("Making train_examples is completed!")
    # dev_examples = make_examples('dev')
    # print("Making dev_examples is completed!")
    # test_examples = make_test_examples()
    # print("Making test_examples is completed!")

    # with open('./datasets/train_examples.pkl', 'wb') as f:
    #     pickle.dump(train_examples, f)
    
    # with open('./datasets/dev_examples.pkl', 'wb') as f:
    #     pickle.dump(dev_examples, f)
    
    # with open('./datasets/test_examples.pkl', 'wb') as f:
    #     pickle.dump(test_examples, f)

    # with open('./datasets/train_examples.pkl', 'rb') as f:
    #     train_examples = pickle.load(f)
    #     print(len(train_examples))

    # with open('./datasets/dev_examples.pkl', 'rb') as f:
    #     dev_examples = pickle.load(f)
    #     print(len(dev_examples))

    # with open('./datasets/test_examples.pkl', 'rb') as f:
    #     test_examples = pickle.load(f)
    #     print(len(test_examples))

    