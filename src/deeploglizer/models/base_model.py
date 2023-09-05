import os
import time
import torch
import logging
import numpy as np
import pandas as pd
from torch import nn
from collections import defaultdict
from sklearn.metrics import (accuracy_score,
                             f1_score,
                             recall_score,
                             precision_score,
                             roc_curve,
                             auc,
                             precision_recall_curve,
                             roc_auc_score,
                             average_precision_score)

from src.deeploglizer.common.utils import set_device, tensor2flatten_arr
from src.deeploglizer.common.lr import PolynomialDecayLR

import warnings
warnings.filterwarnings('ignore')

class Embedder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        pretrain_matrix=None,
        freeze=False,
        feature_type='semantics',
        embedding_type='bert',
    ):
        super(Embedder, self).__init__()
        self.feature_type = feature_type
        self.embedding_type = embedding_type
        if self.embedding_type == 'bert':
            self.linear = nn.Linear(768, embedding_dim)
        else:
            if pretrain_matrix is not None:
                self.embedding_layer = nn.Embedding.from_pretrained(
                    pretrain_matrix, padding_idx=1, freeze=freeze
                )
            else:
                self.embedding_layer = nn.Embedding(
                    vocab_size, embedding_dim, padding_idx=1
                )

    def forward(self, x):
        if self.feature_type == 'semantics':
            if self.embedding_type == 'bert':
                return self.linear(x.float())
            elif self.embedding_type == 'tfidf':
                return torch.matmul(x, self.embedding_layer.weight.float())
        else:
            return self.embedding_layer(x.long())


class ForcastBasedModel(nn.Module):
    def __init__(
        self,
        meta_data,
        encoder_type,
        hidden_size,
        kernel_sizes,
        num_directions,
        model_save_path,
        feature_type,
        label_type,
        eval_type,
        topk,
        embedding_type,
        embedding_dim,
        freeze=False,
        gpu=-1,
        anomaly_ratio=None,
        patience=3,
        batch_size=64,
        warmup_epoch=3,
        peak_lr=1e-2,
        end_lr=2e-4,
        weight_decay=1e-4,
        **kwargs,
    ):
        super(ForcastBasedModel, self).__init__()
        self.device = set_device(gpu)
        self.topk = topk
        self.meta_data = meta_data
        self.encoder_type = encoder_type
        self.feature_type = feature_type
        self.label_type = label_type
        self.eval_type = eval_type
        self.anomaly_ratio = anomaly_ratio  # only used for auto encoder
        self.patience = patience
        self.batch_size = batch_size
        self.warmup_epoch = warmup_epoch
        self.peak_lr = peak_lr
        self.end_lr = end_lr
        self.weight_decay = weight_decay
        self.time_tracker = {}
        self.hidden_size = hidden_size
        self.kernel_sizes = kernel_sizes
        self.num_directions = num_directions

        if self.encoder_type == 'cnn':
            self.center = torch.zeros(hidden_size * len(kernel_sizes)).to(self.device)
        elif self.encoder_type == 'linear':
            self.center = torch.zeros(1, device=self.device)
        elif self.encoder_type == 'lstm':
            self.center = torch.zeros(self.hidden_size * self.num_directions, device=self.device)
        elif self.encoder_type in ['gin', 'gine']:
            self.center = torch.zeros(self.hidden_size, device=self.device)
        else:
            RuntimeError("The specified network is not supported!")

        self.radius = torch.tensor(0).to(self.device)

        os.makedirs(model_save_path, exist_ok=True)
        self.model_save_file = os.path.join(model_save_path, "model.ckpt")
        if self.encoder_type in ['linear', 'lstm', 'cnn']:
            if feature_type in ["sequentials", "semantics", "quantitatives"]:
                self.embedder = Embedder(
                    meta_data["vocab_size"],
                    embedding_dim=embedding_dim,
                    pretrain_matrix=meta_data.get("pretrain_matrix", None),
                    freeze=freeze,
                    feature_type=feature_type,
                    embedding_type=embedding_type
                )
            else:
                logging.info(f'Unrecognized feature type, except sequentials or semantics or quantitatives, got {feature_type}')

    def evaluate(self, test_loader, dtype="test"):
        logging.info("Evaluating {} data.".format(dtype))

        if self.label_type == "next_log":
            return self.__evaluate_next_log(test_loader, dtype=dtype)
        elif self.label_type == "anomaly":
            return self.__evaluate_anomaly(test_loader, dtype=dtype)
        elif self.label_type == "none":
            return self.__evaluate_recst(test_loader, dtype=dtype)

    def __evaluate_recst(self, test_loader, dtype="test"):
        self.eval()  # set to evaluation mode
        with torch.no_grad():
            y_pred = []
            store_dict = defaultdict(list)
            infer_start = time.time()
            for idx, batch_input in enumerate(test_loader):
                return_dict = self.forward(self.__input2device(batch_input))
                y_pred = return_dict["y_pred"]

                if self.encoder_type in ['linear', 'lstm', 'cnn']:
                    if self.eval_type == "session":
                        store_dict["session_idx"].extend(tensor2flatten_arr(batch_input["session_idx"]))
                    else:
                        store_dict["session_idx"].extend(
                            tensor2flatten_arr(torch.arange(idx * y_pred.size(0), (idx + 1) * y_pred.size(0)))
                        )
                    store_dict["window_anomalies"].extend(
                        tensor2flatten_arr(batch_input["window_anomalies"])
                    )
                else:
                    store_dict["session_idx"].extend(
                        tensor2flatten_arr(torch.arange(idx * y_pred.size(0), (idx + 1) * y_pred.size(0)))
                    )
                    store_dict["window_anomalies"].extend(
                        tensor2flatten_arr(batch_input["y"])
                    )
                store_dict["window_preds"].extend(tensor2flatten_arr(y_pred))
            infer_end = time.time()
            logging.info("Finish inference [{:.2f}s]".format(infer_end - infer_start))
            self.time_tracker["test"] = infer_end - infer_start

            store_df = pd.DataFrame(store_dict)

            use_cols = ["session_idx", "window_anomalies", "window_preds"]
            session_df = (
                store_df[use_cols]
                .groupby("session_idx", as_index=False)
                .max()  # most anomalous window
            )

            eval_results, pred_labels = seek_best_result(session_df)
            return eval_results

    def __evaluate_anomaly(self, test_loader, dtype="test"):

        self.eval()  # set to evaluation mode
        with torch.no_grad():
            y_pred = []
            store_dict = defaultdict(list)
            infer_start = time.time()
            for idx, batch_input in enumerate(test_loader):
                return_dict = self.forward(self.__input2device(batch_input))
                y_prob, y_pred = return_dict["y_pred"].max(dim=1)

                if self.eval_type == "session":
                    store_dict["session_idx"].extend(tensor2flatten_arr(batch_input["session_idx"]))
                else:
                    store_dict["session_idx"].extend(
                        tensor2flatten_arr(torch.arange(idx * y_pred.size(0), (idx+1) * y_pred.size(0)))
                    )
                store_dict["window_anomalies"].extend(
                    tensor2flatten_arr(batch_input["window_anomalies"])
                )
                store_dict["window_preds"].extend(tensor2flatten_arr(y_pred))
            infer_end = time.time()
            logging.info("Finish inference. [{:.2f}s]".format(infer_end - infer_start))
            self.time_tracker["test"] = infer_end - infer_start

            store_df = pd.DataFrame(store_dict)
            use_cols = ["session_idx", "window_anomalies", "window_preds"]
            session_df = store_df[use_cols].groupby("session_idx", as_index=False).sum()
            pred = (session_df[f"window_preds"] > 0).astype(int)
            y = (session_df["window_anomalies"] > 0).astype(int)

            fpr_ab, tpr_ab, _ = roc_curve(y, pred)

            eval_results = {
                "f1": f1_score(y, pred),
                "rc": recall_score(y, pred),
                "pc": precision_score(y, pred),
                "roc": auc(fpr_ab, tpr_ab),
                "acc": accuracy_score(y, pred),
            }
            logging.info({k: f"{v:.5f}" for k, v in eval_results.items()})
            return eval_results

    def __evaluate_next_log(self, test_loader, dtype="test"):
        model = self.eval()  # set to evaluation mode
        with torch.no_grad():
            y_pred = []
            store_dict = defaultdict(list)
            infer_start = time.time()
            for idx, batch_input in enumerate(test_loader):
                return_dict = model.forward(self.__input2device(batch_input))
                y_pred = return_dict["y_pred"]
                y_prob_topk, y_pred_topk = torch.topk(y_pred, self.topk)  # b x topk

                if self.eval_type == "session":
                    store_dict["session_idx"].extend(
                        tensor2flatten_arr(batch_input["session_idx"])
                    )
                else:
                    store_dict["session_idx"].extend(
                        tensor2flatten_arr(torch.arange(idx * y_pred.size(0), (idx+1) * y_pred.size(0)))
                    )
                store_dict["window_anomalies"].extend(
                    tensor2flatten_arr(batch_input["window_anomalies"])
                )
                store_dict["window_labels"].extend(
                    tensor2flatten_arr(batch_input["window_labels"])
                )
                store_dict["x"].extend(batch_input["features"].data.cpu().numpy())
                store_dict["y_pred_topk"].extend(y_pred_topk.data.cpu().numpy())
                store_dict["y_prob_topk"].extend(y_prob_topk.data.cpu().numpy())
            infer_end = time.time()
            logging.info("Finish inference. [{:.2f}s]".format(infer_end - infer_start))
            self.time_tracker["test"] = infer_end - infer_start
            store_df = pd.DataFrame(store_dict)
            best_result = None
            best_f1 = -float("inf")

            count_start = time.time()

            topkdf = pd.DataFrame(store_df["y_pred_topk"].tolist())
            logging.info("Calculating acc sum.")
            hit_df = pd.DataFrame()
            for col in sorted(topkdf.columns):
                topk = col + 1
                hit = (topkdf[col] == store_df["window_labels"]).astype(int)
                hit_df[topk] = hit
                if col == 0:
                    acc_sum = 2 ** topk * hit
                else:
                    acc_sum += 2 ** topk * hit
            acc_sum[acc_sum == 0] = 2 ** (1 + len(topkdf.columns))
            hit_df["acc_num"] = acc_sum

            for col in sorted(topkdf.columns):
                topk = col + 1
                check_num = 2 ** topk
                store_df["window_pred_anomaly_{}".format(topk)] = (
                    ~(hit_df["acc_num"] <= check_num)
                ).astype(int)
            # store_df.to_csv("store_{}_2.csv".format(dtype), index=False)

            logging.info("Finish generating store_df.")

            if self.eval_type == "session":
                use_cols = ["session_idx", "window_anomalies"] + [
                    f"window_pred_anomaly_{topk}" for topk in range(1, self.topk + 1)
                ]
                session_df = (
                    store_df[use_cols].groupby("session_idx", as_index=False).sum()
                )
            else:
                session_df = store_df
            # session_df.to_csv("session_{}_2.csv".format(dtype), index=False)

            for topk in range(1, self.topk + 1):
                pred = (session_df[f"window_pred_anomaly_{topk}"] > 0).astype(int)
                y = (session_df["window_anomalies"] > 0).astype(int)
                window_topk_acc = 1 - store_df["window_anomalies"].sum() / len(store_df)
                eval_results = {
                    "f1": f1_score(y, pred),
                    "rc": recall_score(y, pred),
                    "pc": precision_score(y, pred),
                    "top{}-acc".format(topk): window_topk_acc,
                }
                logging.info({k: f"{v:.4f}" for k, v in eval_results.items()})
                if eval_results["f1"] >= best_f1:
                    best_result = eval_results
                    best_f1 = eval_results["f1"]
            count_end = time.time()
            logging.info("Finish counting [{:.2f}s]".format(count_end - count_start))
            return best_result

    def __input2device(self, batch_input):
        return {k: v.to(self.device) for k, v in batch_input.items()}

    def save_model(self):
        logging.info("Saving model to {}".format(self.model_save_file))
        try:
            torch.save(
                self.state_dict(),
                self.model_save_file,
                _use_new_zipfile_serialization=False,
            )
        except:
            torch.save(self.state_dict(), self.model_save_file)

    def load_model(self, model_save_file=""):
        logging.info("Loading model from {}".format(self.model_save_file))
        self.load_state_dict(torch.load(model_save_file, map_location=self.device))

    def fit(self, train_loader, validation_loader=None, test_loader=None, epoches=10):
        self.to(self.device)
        logging.info(
            "Start training on {} batches with {}.".format(
                len(train_loader), self.device
            )
        )
        best_f1 = -float("inf")
        best_results = None
        worse_count = 0

        for epoch in range(1, epoches + 1):
            epoch_time_start = time.time()
            model = self.train()
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                          lr=self.peak_lr, weight_decay=self.weight_decay)
            scheduler = PolynomialDecayLR(optimizer,
                                          warmup_updates=self.warmup_epoch * self.batch_size,
                                          tot_updates=epoches * self.batch_size,
                                          lr=self.peak_lr,
                                          end_lr=self.end_lr,
                                          power=1.0)

            batch_cnt = 0
            epoch_loss = 0

            self.update_center(train_loader)

            for batch_input in train_loader:
                loss = model.forward(self.__input2device(batch_input))["loss"]
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                epoch_loss += loss.item()
                batch_cnt += 1
            epoch_loss = epoch_loss / batch_cnt

            epoch_time_elapsed = time.time() - epoch_time_start
            self.time_tracker["train"] = epoch_time_elapsed

            logging.info(
                    "Epoch {}/{}, training loss: {:.5f} [{:.2f}s]".format(epoch, epoches, epoch_loss, epoch_time_elapsed)
            )

            if validation_loader is not None:
                testing_time = time.time()
                eval_results = self.evaluate(validation_loader, dtype="validation")
                self.time_tracker["validation"] = time.time() - testing_time
            else:
                testing_time = time.time()
                eval_results = self.evaluate(test_loader, dtype="test")
                self.time_tracker["test"] = time.time() - testing_time

            if eval_results["f1"] > best_f1:
                best_f1 = eval_results["f1"]
                best_results = eval_results
                best_results["converge"] = int(epoch)
                self.save_model()
                worse_count = 0
            else:
                worse_count += 1
                if worse_count >= self.patience:
                    logging.info("Early stop at epoch: {}".format(epoch))
                    break

        if validation_loader is not None:
            testing_time = time.time()
            self.load_model(self.model_save_file)
            best_results = self.evaluate(test_loader)
            self.time_tracker["test"] = time.time() - testing_time

        return best_results

    def update_center(self, train_loader):
        model = self.eval()
        if self.encoder_type == 'cnn':
            results = {"n_center": torch.zeros(self.hidden_size*len(self.kernel_sizes), device=self.device),
                       "r_loss": 0.0, "p_loss": 0.0, "o_loss": 0.0}
        elif self.encoder_type == 'linear':
            results = {"n_center": torch.zeros(1, device=self.device), "r_loss": 0.0, "p_loss": 0.0, "o_loss": 0.0}
        elif self.encoder_type == 'lstm':
            results = {"n_center": torch.zeros(self.hidden_size * self.num_directions, device=self.device), "r_loss": 0.0, "p_loss": 0.0, "o_loss": 0.0}
        elif self.encoder_type in ['gin', 'gine']:
            results = {"n_center": torch.zeros(self.hidden_size, device=self.device), "r_loss": 0.0, "p_loss": 0.0, "o_loss": 0.0}
        else:
            RuntimeError("The specified network is not supported!")
        n_samples = 0
        radius = torch.tensor(0, device=self.device)
        with torch.no_grad():
            for batch_input in train_loader:
                ret = model.forward(self.__input2device(batch_input))
                for k, v in ret.items():
                    n_samples += results['n_center'].size(0)
                    if k == 'n_center':
                        results[k] += torch.sum(v, dim=0)
                        cur_radius = torch.max(torch.sum((v - self.center) ** 2, dim=-1))
                        if cur_radius > radius:
                            radius = cur_radius
                    elif k in ['r_loss', 'p_loss', 'o_loss']:
                        results[k] += v.item()

        for k, v in results.items():
            results[k] /= n_samples
        self.center = results['n_center']
        self.radius = radius

        print(f"The radius = {radius.item()}, The rect loss = {results['r_loss']}, The project loss = {results['p_loss']}, The dist loss = {results['o_loss']}")

def seek_best_result(session_df):
    best_results = {
        "f1": 0.0,
        "rc": 0.0,
        "pc": 0.0,
        "prc": 0.0,
        "roc": 0.0,
        "apc": 0.0,
        "acc": 0.0,
    }
    best_anomaly_ratio = 0.0
    best_pred = []

    for anomaly_ratio in np.arange(0.001, 0.5, 0.001):
        threshold = np.percentile(
            session_df[f"window_preds"].values, 100 - round(anomaly_ratio, 2) * 100
        )
        pred = (session_df[f"window_preds"] > threshold).astype(int)
        y = (session_df["window_anomalies"] > 0).astype(int)

        roc_auc = roc_auc_score(y, session_df[f"window_preds"].to_numpy())

        apc_score = average_precision_score(y, session_df[f"window_preds"].to_numpy())

        precision, recall, _ = precision_recall_curve(y, session_df[f"window_preds"].to_numpy())

        eval_results = {
            "f1": f1_score(y, pred),
            "rc": recall_score(y, pred),
            "pc": precision_score(y, pred),
            "prc": auc(recall, precision),
            "roc": roc_auc,
            "apc": apc_score,
            "acc": accuracy_score(y, pred),
        }
        if eval_results['f1'] > best_results['f1']:
            best_results = eval_results.copy()
            best_anomaly_ratio = round(anomaly_ratio, 5)
            best_pred = np.array(pred.values.tolist())

    logging.info({k: f"{v:.4f}" for k, v in best_results.items()})
    logging.info({f"best anomaly ratio setting: {best_anomaly_ratio}"})
    return best_results, best_pred
