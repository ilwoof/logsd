"""
The interface to load log datasets. The datasets currently supported include
HDFS and BGL.

Authors:
    LogPAI Team

"""

import math
import logging
import pandas as pd
import itertools
import os
import numpy as np
import re
import pickle
import json
from collections import OrderedDict, defaultdict, Counter
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedShuffleSplit

from src.deeploglizer.common.utils import decision


def load_sessions(data_dir,
                  without_duplicate=False,
                  session_size=100,
                  sampling_size=1.0,
                  do_padding=False,
                  session_level='entry',
                  semi_supervised=False,
                  no_normal_with_unseen=False,
                  show_statistics=False):
    if without_duplicate:
        with open(os.path.join(data_dir, f"data_desc_wo_duplicate_{session_size}{session_level}.json"), "r") as fr:
            data_desc = json.load(fr)
        with open(os.path.join(data_dir, f"session_train_wo_duplicate_{session_size}{session_level}.pkl"), "rb") as fr:
            session_train = pickle.load(fr)
        with open(os.path.join(data_dir, f"session_test_wo_duplicate_{session_size}{session_level}.pkl"), "rb") as fr:
            session_test = pickle.load(fr)
    else:
        with open(os.path.join(data_dir, f"data_desc_{session_size}{session_level}.json"), "r") as fr:
            data_desc = json.load(fr)
        with open(os.path.join(data_dir, f"session_train_{session_size}{session_level}.pkl"), "rb") as fr:
            session_train = pickle.load(fr)
        with open(os.path.join(data_dir, f"session_test_{session_size}{session_level}.pkl"), "rb") as fr:
            session_test = pickle.load(fr)

    if show_statistics:
        file_name = data_desc.get('log_file', 0)
        window_size = data_desc.get('window_range', 0)
        if 'BGL' in file_name:
            dataset = 'bgl'
            print(f"The dataset is BGL-{window_size}")
        elif 'Spirit' in file_name:
            dataset = 'spirit'
            print(f"The dataset is Spirit-{window_size}")
        elif 'Thunderbird' in file_name:
            dataset = 'tbd'
            print(f"The dataset is Tbd-{window_size}")
        elif 'HDFS' in file_name:
            dataset = 'hdfs'
            print(f"The dataset is HDFS-session")
        else:
            RuntimeError("The specified dataset cannot be analysed")
            exit(-1)

        total_train_ulog = set(itertools.chain(*[v["templates"] for k, v in session_train.items()]))

        if dataset == 'hdfs':
            normal_train_ulog = set(itertools.chain(*[v["templates"] for k, v in session_train.items() if v["label"] == 0]))
        else:
            normal_train_ulog = set(itertools.chain(*[v["templates"] for k, v in session_train.items() if int(sum(v["label"])) == 0]))

        test_ulog = set(itertools.chain(*[v["templates"] for k, v in session_test.items()]))

        ulog_new_under_un = test_ulog - total_train_ulog
        ulog_new_under_semi = test_ulog - normal_train_ulog
        ulog_union_under_un = total_train_ulog.copy()
        ulog_union_under_un.update(test_ulog)
        ulog_union_under_semi = normal_train_ulog.copy()
        ulog_union_under_semi.update(test_ulog)

        semi_normal_with_unseen_num = 0
        un_normal_with_unseen_num = 0
        semi_anomaly_with_unseen_num = 0
        un_anomaly_with_unseen_num = 0
        total_test_instance_num = 0
        total_test_anomaly_num = 0

        normal_unseen_templates = []
        unsu_unseen_normal_event_dict = Counter()
        unsu_unseen_abnormal_event_dict = Counter()
        semi_unseen_normal_event_dict = Counter()
        semi_unseen_abnormal_event_dict = Counter()

        strange_seen_abnormal = Counter()

        for k, v in session_test.items():
            instance = set(v["templates"])
            instance_anomalous_flag = v["label"] if not isinstance(v["label"], list) else int(sum(v["label"]) > 0)

            # df = pd.DataFrame(total_train_ulog, columns=['templates'])
            # df.to_csv(f"./experiment_records/{dataset}{window_size}_total_train_templates.csv", index=True)
            #
            # df = pd.DataFrame(normal_train_ulog, columns=['templates'])
            # df.to_csv(f"./experiment_records/{dataset}{window_size}_normal_train_templates.csv", index=True)

            total_test_instance_num += 1
            if instance_anomalous_flag == 1:
                total_test_anomaly_num += 1
            if len(instance - total_train_ulog) > 0:
                if instance_anomalous_flag == 0:
                    un_normal_with_unseen_num += 1
                    unsu_unseen_normal_event_dict += Counter(instance - total_train_ulog)
                else:
                    un_anomaly_with_unseen_num += 1
                    unsu_unseen_abnormal_event_dict += Counter(instance - total_train_ulog)
            if len(instance - normal_train_ulog) > 0:
                if instance_anomalous_flag == 0:
                    semi_normal_with_unseen_num += 1
                    normal_unseen_templates.append(",".join(list(instance - normal_train_ulog)))
                    semi_unseen_normal_event_dict += Counter(instance - normal_train_ulog)
                else:
                    semi_anomaly_with_unseen_num += 1
                    semi_unseen_abnormal_event_dict += Counter(instance - normal_train_ulog)
            else:
                if instance_anomalous_flag == 1:
                    strange_seen_abnormal = Counter(instance)
                    print("The strange instance is abnormal but without unseen events")

        # if len(unsu_unseen_normal_event_dict) > 0:
        #     df = pd.DataFrame.from_dict(unsu_unseen_normal_event_dict, orient='index', columns=['frequency'])
        #     df.index.name = 'events'
        #     df.to_csv(f"./experiment_records/{dataset}{window_size}_unsupervised_normal_unseen_templates.csv", index=True)
        #
        # if len(unsu_unseen_abnormal_event_dict) > 0:
        #     df = pd.DataFrame.from_dict(unsu_unseen_abnormal_event_dict, orient='index', columns=['frequency'])
        #     df.index.name = 'events'
        #     df.to_csv(f"./experiment_records/{dataset}{window_size}_unsupervised_abnormal_unseen_templates.csv", index=True)
        #
        # if len(semi_unseen_normal_event_dict) > 0:
        #     df = pd.DataFrame.from_dict(semi_unseen_normal_event_dict, orient='index', columns=['frequency'])
        #     df.index.name = 'events'
        #     df.to_csv(f"./experiment_records/{dataset}{window_size}_semisupervised_normal_unseen_templates.csv", index=True)
        #
        # if len(semi_unseen_abnormal_event_dict) > 0:
        #     df = pd.DataFrame.from_dict(semi_unseen_abnormal_event_dict, orient='index', columns=['frequency'])
        #     df.index.name = 'events'
        #     df.to_csv(f"./experiment_records/{dataset}{window_size}_semisupervised_abnormal_unseen_templates.csv", index=True)
        #
        # if len(strange_seen_abnormal) > 0:
        #     df = pd.DataFrame.from_dict(strange_seen_abnormal, orient='index', columns=['frequency'])
        #     df.index.name = 'events'
        #     df.to_csv(f"./experiment_records/{dataset}{window_size}_strange_abnormal_seen_templates.csv", index=True)

        print(f"Under semi-supervised/un-supervised cases: \n"
              f"    Total events = {len(ulog_union_under_semi)}/{len(ulog_union_under_un)}\n"
              f"    Training set events = {len(normal_train_ulog)}/{len(total_train_ulog)}\n"
              f"    Test set events = {len(test_ulog)}\n"
              f"    Test set unseen events = {len(ulog_new_under_semi)}/{len(ulog_new_under_un)}\n"
              f"    Test set total instances = {total_test_instance_num}\n"
              f"    Test set total abnormal instances = {total_test_anomaly_num}\n"
              f"    Test set abnormal instances with unseen events = {semi_anomaly_with_unseen_num}/{un_anomaly_with_unseen_num}\n"
              f"    Test set abnormal instances without unseen events = {total_test_anomaly_num - semi_anomaly_with_unseen_num}/{total_test_anomaly_num - un_anomaly_with_unseen_num}\n"
              f"    Test set normal instances with unseen events = {semi_normal_with_unseen_num}/{un_normal_with_unseen_num}")

    if no_normal_with_unseen:
        file_name = data_desc.get('log_file', 0)
        total_train_ulog = set(itertools.chain(*[v["templates"] for k, v in session_train.items()]))
        if 'HDFS' in file_name:
            normal_train_ulog = set(itertools.chain(*[v["templates"] for k, v in session_train.items() if v["label"] == 0]))
        else:
            normal_train_ulog = set(itertools.chain(*[v["templates"] for k, v in session_train.items() if int(sum(v["label"])) == 0]))

        remove_list = []
        for k, v in session_test.items():
            label = v["label"] if not isinstance(v["label"], list) else int(sum(v["label"]) > 0)
            if semi_supervised:
                if label == 0 and len(set(v["templates"]) - normal_train_ulog) > 0:
                    remove_list.append(k)
            else:
                if label == 0 and len(set(v["templates"]) - total_train_ulog) > 0:
                    remove_list.append(k)
        session_test = {k: v for k, v in session_test.items() if k not in remove_list}
        del remove_list

    max_template_size = data_desc.get('max_template_size', 0)

    if (session_size == 'session') and (not max_template_size == 0) and do_padding:
        for v in session_train.values():
            v["templates"].extend(["PADDING"] * (max_template_size - len(v["templates"])))

        for v in session_test.values():
            v["templates"].extend(["PADDING"] * (max_template_size - len(v["templates"])))

    train_labels = [
        v["label"] if not isinstance(v["label"], list) else int(sum(v["label"]) > 0)
        for _, v in session_train.items()
    ]
    test_labels = [
        v["label"] if not isinstance(v["label"], list) else int(sum(v["label"]) > 0)
        for _, v in session_test.items()
    ]
    if not sampling_size == 1.0:
        if (sampling_size > 0.0) and (sampling_size < 1.0):
            sampling_size = sampling_size
        elif (sampling_size > 1.0) and (math.ceil(sampling_size) <= len(session_train)):
            sampling_size = math.ceil(sampling_size)
        elif sampling_size > len(session_train):
            sampling_size = len(session_train)
        else:
            raise RuntimeError("The invalid sampling_size")
        if sampling_size < len(session_train):
            kfd = StratifiedShuffleSplit(n_splits=1, train_size=sampling_size, random_state=42)
            key_list = list(session_train.keys())
            for indices, _ in kfd.split(key_list, train_labels):
                session_train = {key_list[i]: session_train[key_list[i]] for i in indices}

    num_train = len(session_train)
    ratio_train = sum(train_labels) / num_train
    num_test = len(session_test)
    ratio_test = sum(test_labels) / num_test
    logging.info("Load from {}".format(data_dir))
    logging.info(json.dumps(data_desc, indent=4))
    logging.info(
        "# train sessions {} ({:.2f} anomalies)".format(num_train, ratio_train)
    )
    logging.info("# test sessions {} ({:.2f} anomalies)".format(num_test, ratio_test))

    if show_statistics:
        dataset_info = []
        dataset_info.append(dataset)
        dataset_info.append(len(ulog_union_under_un))
        dataset_info.append(session_size)
        dataset_info.append(num_train)
        dataset_info.append(len(total_train_ulog))
        dataset_info.append(sum(train_labels))
        dataset_info.append(f"{ratio_train:.4f}")
        dataset_info.append(num_test)
        dataset_info.append(len(test_ulog))
        dataset_info.append(sum(test_labels))
        dataset_info.append(f"{ratio_test:.4f}")
        save_dataset_statistic_info(dataset_info)
        print(f"Statistics info has been collected and saved.")
        exit(-1)

    return session_train, session_test, max_template_size


class log_dataset(Dataset):
    def __init__(self, session_dict, feature_type="semantics"):
        flatten_data_list = []
        # flatten all sessions
        for session_idx, data_dict in enumerate(session_dict.values()):
            features = data_dict["features"][feature_type]
            window_labels = data_dict["window_labels"]
            window_anomalies = data_dict["window_anomalies"]
            for window_idx in range(len(window_labels)):
                sample = {
                    "session_idx": session_idx,  # not session id
                    "features": features[window_idx],
                    "window_labels": window_labels[window_idx],
                    "window_anomalies": window_anomalies[window_idx],
                }
                flatten_data_list.append(sample)
        self.flatten_data_list = flatten_data_list

    def __len__(self):
        return len(self.flatten_data_list)

    def __getitem__(self, idx):
        return self.flatten_data_list[idx]

def load_BGL(
    log_file,
    train_ratio=None,
    test_ratio=0.8,
    train_anomaly_ratio=0,
    random_partition=False,
    filter_normal=True,
    **kwargs
):
    logging.info("Loading BGL logs from {}.".format(log_file))
    struct_log = pd.read_csv(log_file, engine="c", na_filter=False, memory_map=True)
    # struct_log.sort_values(by=["Timestamp"], inplace=True)
    logging.info("{} lines loaded.".format(struct_log.shape[0]))

    templates = struct_log["EventTemplate"].values
    labels = struct_log["Label"].map(lambda x: x != "-").astype(int).values

    total_indice = np.array(list(range(templates.shape[0])))
    if random_partition:
        logging.info("Using random partition.")
        np.random.shuffle(total_indice)

    if train_ratio is None:
        train_ratio = 1 - test_ratio
    assert train_ratio + test_ratio <= 1, "train_ratio + test_ratio should <= 1."
    train_lines = int(train_ratio * len(total_indice))
    test_lines = int(test_ratio * len(total_indice))

    idx_train = total_indice[0:train_lines]
    idx_test = total_indice[-test_lines:]

    idx_train = [
        idx
        for idx in idx_train
        if (labels[idx] == 0 or (labels[idx] == 1 and decision(train_anomaly_ratio)))
    ]

    if filter_normal:
        logging.info(
            "Filtering unseen normal templates in {} test data.".format(len(idx_test))
        )
        seen_normal = set(templates[idx_train].tolist())
        idx_test = [
            idx
            for idx in idx_test
            if not (labels[idx] == 0 and (templates[idx] not in seen_normal))
        ]

    session_train = {
        "all": {"templates": templates[idx_train].tolist(), "label": labels[idx_train]}
    }
    session_test = {
        "all": {"templates": templates[idx_test].tolist(), "label": labels[idx_test]}
    }

    labels_train = labels[idx_train]
    labels_test = labels[idx_test]

    train_anomaly = 100 * sum(labels_train) / len(labels_train)
    test_anomaly = 100 * sum(labels_test) / len(labels_test)

    logging.info("# train lines: {} ({:.2f}%)".format(len(labels_train), train_anomaly))
    logging.info("# test lines: {} ({:.2f}%)".format(len(labels_test), test_anomaly))

    return session_train, session_test


def load_HDFS(
    log_file,
    label_file,
    train_ratio=None,
    test_ratio=None,
    train_anomaly_ratio=1,
    random_partition=False,
    **kwargs
):
    """Load HDFS structured log into train and test data

    Arguments
    ---------
        TODO

    Returns
    -------
        TODO
    """
    logging.info("Loading HDFS logs from {}.".format(log_file))
    struct_log = pd.read_csv(log_file, engine="c", na_filter=False, memory_map=True)
    # struct_log.sort_values(by=["Date", "Time"], inplace=True)

    # assign labels
    label_data = pd.read_csv(label_file, engine="c", na_filter=False, memory_map=True)
    label_data["Label"] = label_data["Label"].map(lambda x: int(x == "Anomaly"))
    label_data_dict = dict(zip(label_data["BlockId"], label_data["Label"]))

    session_dict = OrderedDict()
    column_idx = {col: idx for idx, col in enumerate(struct_log.columns)}
    for _, row in enumerate(struct_log.values):
        blkId_list = re.findall(r"(blk_-?\d+)", row[column_idx["Content"]])
        blkId_set = set(blkId_list)
        for blk_Id in blkId_set:
            if blk_Id not in session_dict:
                session_dict[blk_Id] = defaultdict(list)
            session_dict[blk_Id]["templates"].append(row[column_idx["EventTemplate"]])

    for k in session_dict.keys():
        session_dict[k]["label"] = label_data_dict[k]

    session_idx = list(range(len(session_dict)))
    # split data
    if random_partition:
        logging.info("Using random partition.")
        np.random.shuffle(session_idx)

    session_ids = np.array(list(session_dict.keys()))
    session_labels = np.array(list(map(lambda x: label_data_dict[x], session_ids)))

    if train_ratio is None:
        train_ratio = 1 - test_ratio
    train_lines = int(train_ratio * len(session_idx))
    test_lines = int(test_ratio * len(session_idx))

    session_idx_train = session_idx[0:train_lines]
    session_idx_test = session_idx[-test_lines:]

    session_id_train = session_ids[session_idx_train]
    session_id_test = session_ids[session_idx_test]
    session_labels_train = session_labels[session_idx_train]
    session_labels_test = session_labels[session_idx_test]

    logging.info("Total # sessions: {}".format(len(session_ids)))

    session_train = {
        k: session_dict[k]
        for k in session_id_train
        if (session_dict[k]["label"] == 0)
        or (session_dict[k]["label"] == 1 and decision(train_anomaly_ratio))
    }

    session_test = {k: session_dict[k] for k in session_id_test}

    session_labels_train = [v["label"] for k, v in session_train.items()]
    session_labels_test = [v["label"] for k, v in session_test.items()]

    train_anomaly = 100 * sum(session_labels_train) / len(session_labels_train)
    test_anomaly = 100 * sum(session_labels_test) / len(session_labels_test)

    logging.info(
        "# train sessions: {} ({:.2f}%)".format(len(session_train), train_anomaly)
    )
    logging.info(
        "# test sessions: {} ({:.2f}%)".format(len(session_test), test_anomaly)
    )

    return session_train, session_test


def load_HDFS_semantic(log_semantic_path):
    train = os.path.join(log_semantic_path, "session_train.pkl")
    test = os.path.join(log_semantic_path, "session_test.pkl")

    with open(train, "rb") as fr:
        session_train = pickle.load(fr)

    with open(test, "rb") as fr:
        session_test = pickle.load(fr)

    # session_test = {k: v for i, (k, v) in enumerate(session_test.items()) if i < 50000}
    logging.info(
        "# train sessions: {}, # test sessions: {}".format(
            len(session_train), len(session_test)
        )
    )
    return session_train, session_test


def load_HDFS_id(log_id_path):
    train = os.path.join(log_id_path, "hdfs_train")
    test_normal = os.path.join(log_id_path, "hdfs_test_normal")
    test_anomaly = os.path.join(log_id_path, "hdfs_test_abnormal")

    session_train = {}
    for idx, line in enumerate(open(train)):
        sample = {"templates": line.split(), "label": 0}
        session_train[idx] = sample

    session_test = {}
    for idx, line in enumerate(open(test_normal)):
        if idx > 50000:
            break
        sample = {"templates": line.split(), "label": 0}
        session_test[idx] = sample

    for idx, line in enumerate(open(test_anomaly), len(session_test)):
        if idx > 100000:
            break
        sample = {"templates": line.split(), "label": 1}
        session_test[idx] = sample

    logging.info(
        "# train sessions: {}, # test sessions: {}".format(
            len(session_train), len(session_test)
        )
    )

    # logging.info("# test sessions: {} ({:.2f}%)".format(len(session_test), test_anomaly_ratio))
    return session_train, session_test

def save_dataset_statistic_info(dataset_info):
    column_title = ['dataset', 'total_event_num', 'window_size', 'train_sequence_num', 'train_event_num', 'train_anomaly_num', 'train_anomaly_ratio', 'test_sequence_num', 'test_event_num', 'test_anomaly_num', 'test_anomaly_ratio']
    result_file_name = './experiment_records/data_statistic.csv'
    if os.path.exists(result_file_name):
        df = pd.read_csv(result_file_name, encoding='utf-8-sig')
        df.loc[len(df)] = dataset_info
        df.to_csv(result_file_name, index=False)
    else:
        pd.DataFrame([dataset_info], columns=column_title).to_csv(result_file_name, index=False, encoding='utf-8-sig')
