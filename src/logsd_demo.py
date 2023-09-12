#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
sys.path.append("../")
import argparse
from torch.utils.data import DataLoader

from deeploglizer.models.ss_network import SS_Net
from deeploglizer.common.preprocess import FeatureExtractor
from deeploglizer.common.dataloader import load_sessions, log_dataset
from deeploglizer.common.utils import seed_everything, dump_params, dump_final_results, metric_list
from sklearn.model_selection import StratifiedShuffleSplit

parser = argparse.ArgumentParser()

# Model params
parser.add_argument("--framework", default="SSD", type=str)
parser.add_argument("--hidden_size", default=128, choices=[32, 64, 128, 256], type=int)

parser.add_argument("--embedding_dim", default=32, type=int)
parser.add_argument("--num_layers", default=2, type=int)
parser.add_argument("--encoder_type", default="cnn", type=str)

parser.add_argument('--reconstruction_mode', default='global2local', type=str, choices=['global', 'local2local', 'global2local'], help='reconstruction paradigm')
parser.add_argument('--masking_by', default='frequency', type=str, choices=['frequency', 'probability', 'no_masking'], help='The masking strategy')
parser.add_argument('--masking_mode', default='row_value_mask', help='Masking mode')
parser.add_argument('--masking_ratio', default=-1.0, type=float, help='The ratio of masking when masking mode is enabled')  # -1.0 means randomly masking range from 0.1 to 0.5
parser.add_argument('--session_level', default='entry', choices=['entry', 'secs'])
parser.add_argument('--loss_ablation', default="all", type=str, choices=["all", "no_pj", 'no_oc', 'no_rect', 'rect_only', 'projection_only', 'oc_only'], help='Ablation loss test')

parser.add_argument('--alpha', default=50.0, type=float, choices=[0, 0.001, 0.01, 0.1, 1, 10, 50, 100], help='The weight for reconstruction loss')
parser.add_argument('--beta', default=1.0, type=float, choices=[0, 0.001, 0.01, 0.1, 1, 10, 50, 100], help='The weight for projection loss')
parser.add_argument('--gamma', default=1.0, type=float, choices=[0, 0.001, 0.01, 0.1, 1, 10, 50, 100], help='The weight for one-class loss')

# CNN model params
parser.add_argument("--kernel_sizes", default=[2, 3, 4], type=int, nargs="+")
parser.add_argument("--decoder_type", default="deconv", type=str)
parser.add_argument("--pooling_mode", default="max", type=str, choices=["max", "mean"])

# Dataset params
parser.add_argument("--datasets", nargs="+", default=['bgl'], choices=['bgl', 'spirit', 'hdfs'], help='The dataset to be processed')
parser.add_argument("--data_dir", default="../data/processed/", type=str)
parser.add_argument("--session_size", nargs="+", default=[100], choices=[100, 60, 20], type=int, help='The target detection size')
parser.add_argument("--window_size", default=100, choices=[100, 60, 20], type=int, help='The actual decomposed size')
parser.add_argument("--stride", default=0, type=int, help='The sliding step size, 0 indicates equal to window size ')
parser.add_argument("--without_duplicate", default=True, help='Use the dataset without consecutive duplicate events')

# Input params
parser.add_argument("--feature_type", default="semantics", type=str)
parser.add_argument("--embedding_type", default="tfidf", type=str)
parser.add_argument("--label_type", default="none", type=str)
parser.add_argument("--pretrain_path", default=None, type=str)
parser.add_argument("--max_token_len", default=50, type=int)
parser.add_argument("--min_token_count", default=1, type=int)

# Uncomment the following to use pretrained word embeddings. The "embedding_dim" should be set as 300
# parser.add_argument(
#     "--pretrain_path", default="../data/pretrain/wiki-news-300d-1M.vec", type=str
# )

# Training params
parser.add_argument("--semi_supervised", default=True)
parser.add_argument("--no_validation", default=True, action="store_true")
parser.add_argument("--train_ratio", default=0.9, type=float)
parser.add_argument("--sampling_size", default=1.0, type=float)
parser.add_argument("--run_times", default=3, type=int)
parser.add_argument("--epoches", default=100, type=int)
parser.add_argument("--batch_size", default=64, type=int)
parser.add_argument("--patience", default=20, type=int)
parser.add_argument("--weight_decay", default=1e-4, type=float)
parser.add_argument("--peak_lr", default=1e-2, type=float)
parser.add_argument("--end_lr", default=1e-4, type=float)
parser.add_argument("--warmup_epoch", default=0, type=int)

#  Data analysis
parser.add_argument("--show_statistics", action="store_true")

# Others
parser.add_argument("--random_seed", default=42, type=int)
parser.add_argument("--gpu", default=0, type=int)
parser.add_argument("--cache", default=False, type=bool)


if __name__ == "__main__":

    params = vars(parser.parse_args())
    seed_everything(params["random_seed"])
    args = parser.parse_args()

    for dataset in args.datasets:
        params['dataset'] = dataset
        params['window_type'] = 'sliding'
        params['eval_type'] = 'session'

        if dataset == 'hdfs':
            session_windows = ['session']
            params['stride'] = 1 if args.stride == 0 else args.stride
        else:
            params['stride'] = params['window_size'] if args.stride == 0 else args.stride
            session_windows = args.session_size

        assert params['window_size'] >= params['stride']

        for window_size in session_windows:
            data_path = os.path.join(args.data_dir, f"{dataset}")

            raw_session_train, raw_session_test, session_size = load_sessions(data_dir=data_path,
                                                                              without_duplicate=args.without_duplicate,
                                                                              session_size=window_size,
                                                                              sampling_size=args.sampling_size,
                                                                              session_level=args.session_level,
                                                                              semi_supervised=args.semi_supervised,
                                                                              show_statistics=args.show_statistics)

            params['session_size'] = session_size if params['window_type'] == 'session' else window_size

            ext = FeatureExtractor(**params)
            session_train = ext.fit_transform(raw_session_train)
            session_test = ext.transform(raw_session_test, datatype="test")

            total_train = log_dataset(session_train, feature_type=params["feature_type"])
            total_labels_train = [s["window_anomalies"] for s in total_train.flatten_data_list]

            dataset_test = log_dataset(session_test, feature_type=params["feature_type"])
            labels_test = [sequence["window_anomalies"] for sequence in dataset_test.flatten_data_list]

            print(f"train anomaly ratio = {np.array(total_labels_train).sum() / len(total_labels_train):.4f}, "
                  f"test anomaly ratio = {np.array(labels_test).sum() / len(labels_test):.4f}")

            dataloader_test = DataLoader(dataset_test, batch_size=params['batch_size'], shuffle=False, pin_memory=True)

            model_save_path = dump_params(params)

            result_list = {k: [] for k in metric_list}

            kfd = StratifiedShuffleSplit(n_splits=args.run_times, train_size=args.train_ratio, random_state=args.random_seed)

            for tr_index, va_index in kfd.split(total_train, total_labels_train):
                if args.no_validation:
                    if args.semi_supervised:
                        dataset_train = [total_train[i] for i in range(len(total_train)) if total_train[i]["window_anomalies"] == 0]
                    else:
                        dataset_train = total_train
                    dataloader_validation = None
                else:
                    if args.semi_supervised:
                        dataset_train = [total_train[index] for index in tr_index if total_train[index]["window_anomalies"] == 0]
                    else:
                        dataset_train = [total_train[index] for index in tr_index]

                    dataset_validation = [total_train[index] for index in va_index if total_train[index]["window_anomalies"] == 0]

                    dataloader_validation = DataLoader(dataset_validation,
                                                       batch_size=params["batch_size"],
                                                       shuffle=True,
                                                       drop_last=True,
                                                       pin_memory=True)

                dataloader_train = DataLoader(dataset_train,
                                              batch_size=params["batch_size"],
                                              shuffle=True,
                                              drop_last=True,
                                              pin_memory=True)

                print(f"total training sequences = {len(total_train)}, total test sequences = {len(dataset_test)}")
                print(f"training batch = {len(dataloader_train)}, "
                      f"actual training sequences = {len(dataloader_train)*params['batch_size']}, "
                      f"test batch = {len(dataloader_test)}, "
                      f"actual test sequences = {len(dataset_test)}"
                      )

                model = SS_Net(meta_data=ext.meta_data, model_save_path=model_save_path, **params)

                eval_results = model.fit(dataloader_train,
                                         validation_loader=dataloader_validation,
                                         test_loader=dataloader_test,
                                         epoches=params["epoches"],
                                         )

                for key, value in eval_results.items():
                    if key in result_list:
                        result_list[key].append(value)
                result_list['trt'].append(model.time_tracker['train'])
                result_list['tst'].append(model.time_tracker['test'])

            final_eval_results = {k: '-' for k in metric_list}
            for key, value in result_list.items():
                if key in final_eval_results:
                    final_eval_results[key] = f"{np.mean(result_list[key]):.4f}±{np.std(result_list[key]):.4f}"
            dump_final_results(params, final_eval_results)
