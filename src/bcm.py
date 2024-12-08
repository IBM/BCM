import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import scipy as sp

import torch
import torch.nn.functional as F

from sklearnex.linear_model import LogisticRegression

from sklearn.base import BaseEstimator
from sklearn.utils.extmath import softmax
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import LabelEncoder
from sklearn.exceptions import ConvergenceWarning
from sklearn.ensemble import RandomForestClassifier


class BaseClassifier:

    def __init__(self, base_weight, base_bias):
        self.base_weight = base_weight.squeeze(0).squeeze(0)
        self.base_bias = base_bias


    @torch.no_grad()
    def get_base_probas(self, features: torch.tensor) -> torch.tensor:
        """
        Computes probability maps for given query features, using the snapshot of the base model right after the
        training. It only computes values for base classes.

        inputs:
            features : shape [batch_size_val, shot, c, h, w]

        returns :
            probas : shape [batch_size_val, shot, num_base_classes_and_bg, h, w]
        """
        logits = torch.einsum('bochw,cC->boChw', features, self.base_weight) + self.base_bias.view(1, 1, -1, 1, 1)
        return torch.softmax(logits, dim=2)


class Classifier:
    def __init__(self, args, base_classifier, n_tasks):
        self.num_base_classes_and_bg = args.num_classes_tr
        self.num_novel_classes = args.num_classes_val
        self.num_classes = self.num_base_classes_and_bg + self.num_novel_classes
        self.n_tasks = n_tasks
        self.shot = args.shot
        self.ensemble = args.ensemble
        self.beta = args.beta
        self.sampling = args.sampling
        self.top_k = args.top_k

        self.base_class_list = args.base_class_list
        self.novel_class_list = args.novel_class_list

        self.base_classifier = base_classifier


    @staticmethod
    def _valid_mean(t, valid_pixels: torch.tensor, dim: int):
        s = (valid_pixels * t).sum(dim=dim)
        return s / (valid_pixels.sum(dim=dim) + 1e-10)

    @torch.no_grad()
    def init_table(self, features_s: torch.tensor, gt_s: torch.tensor) -> None:
        """
        inputs:
            features_s : shape [num_novel_classes, shot, c, h, w]
            gt_s : shape [num_novel_classes, shot, H, W]
        """
        # Downsample support masks
        ds_gt_s = F.interpolate(gt_s.float(), size=features_s.shape[-2:], mode='nearest')
        ds_gt_s = ds_gt_s.long().unsqueeze(2)  # [n_novel_classes, shot, 1, h, w]

        nc, shot, c, h, w = features_s.shape
        H, W = gt_s.shape[2:]

        base_probas = self.base_classifier.get_base_probas(features_s)
        bpm2, bp2 = base_probas.topk(2, dim=2)
        base_probas_max, base_pred = bpm2[:, :, [0], :, :], bp2[:, :, [0], :, :]
        base_probas_max2, base_pred2 = bpm2[:, :, [1], :, :], bp2[:, :, [1], :, :]

        self.table = {k:[] for k in range(self.num_base_classes_and_bg)}
        self.table_cnt = {k:0 for k in range(self.num_base_classes_and_bg)}

        self.reverse_table = {v: [] for v in range(self.num_base_classes_and_bg, self.num_classes)}
        cnt_table = np.zeros((self.num_base_classes_and_bg, self.num_classes - self.num_base_classes_and_bg),
                             dtype=int)

        for cls in range(self.num_base_classes_and_bg, self.num_classes):
            novel_mask = (ds_gt_s == cls)

            detected_classes, counts = base_pred[novel_mask].unique(return_counts=True)
            for i, dc in enumerate(detected_classes.cpu()):
                cnt_table[dc.item(), cls - self.num_base_classes_and_bg] = counts[i].item()
                self.reverse_table[cls].append((dc.item(), counts[i].item()))
                self.table[dc.item()].append(cls)
                self.table_cnt[dc.item()] += counts[i].item()
                val = base_probas_max[novel_mask][base_pred[novel_mask]==dc]

        top_k = self.top_k
        for k in list(self.reverse_table.keys()):
            self.reverse_table[k] = sorted(self.reverse_table[k], key=lambda x: x[1], reverse=True)
            if top_k != -1:
                self.reverse_table[k] = self.reverse_table[k][:top_k]

        self.table = {k:[] for k in range(self.num_base_classes_and_bg)}
        for k, vv in self.reverse_table.items():
            for v, c in vv:
                self.table[v].append(k)

        bg_prototype = self._valid_mean(features_s, ds_gt_s == 0, (0, 1, 3, 4))

        for k, v in list(self.table.items()):
            if len(v) > 0:
                self.table[k] = torch.as_tensor(v)
            else:
                self.table.pop(k)

        base_probas = F.interpolate(base_probas.view(nc*shot, self.num_base_classes_and_bg, h, w),
                                    size=(H, W), mode='bilinear', align_corners=True)
        _, base_pred = base_probas.max(dim=1)
        base_pred = base_pred.reshape((nc, shot, H, W))
        base_probas = base_probas.reshape((nc, shot, self.num_base_classes_and_bg, H, W))

        for s in range(shot):
            val = np.unique(gt_s[:, s, :, :].cpu())

        _gt_s = gt_s.clone()
        gt_s = {}
        for k, v in self.table.items():
            gt_s[k] = torch.zeros_like(_gt_s)
            new_index = 1
            for class_index in _gt_s.unique().cpu():
                index = (_gt_s == class_index)
                if class_index == 0 or class_index == 255:
                    gt_s[k][index] = class_index
                elif class_index in v:
                    gt_s[k][index] = new_index
                    new_index += 1

        ds_gt_s = {k: F.interpolate(v.float(), size=features_s.size()[-2:], mode='nearest').long()
                   for k, v in gt_s.items()}
        self.ds_gt_s = {k: v.view(nc*shot, h, w) for k, v in ds_gt_s.items()}

        bg_base_class_list = [0] + self.base_class_list
        for k in self.table.keys():
            v = self.table[k] - len(self.base_class_list) - 1


    def preproc(self, X):
        if self.beta != 1.0:
            X = np.power(X, self.beta)
        return X


    def train(self, features_s):
        sk_model, scaler = {}, {}
        n_splits, Cs = 5, 10
        n_C = Cs
        n, _, _, h, w = features_s.shape

        idx_shot = torch.arange(self.shot)
        val_scores = {}

        for k, v in self.table.items():
            val_scores[k] = np.zeros((self.shot, len(v)+1))
            ds_gt_s = self.ds_gt_s[k].view((n, self.shot, h, w))

            sk_model[k] = []
            for i_shot in range(self.shot): # shot-wise ensemble learning
                if self.shot == 1 or not self.ensemble:
                    break

                _features_s = features_s[:, [i_shot], :, :, :]
                _ds_gt_s = ds_gt_s[:, [i_shot], :, :]
                base_probas = self.base_classifier.get_base_probas(_features_s)
                bpm2, bp2 = base_probas.topk(2, dim=2)
                base_probas_max, base_pred = bpm2[:, :, 0, :, :], bp2[:, :, 0, :, :]
                base_probas_max2, base_pred2 = bpm2[:, :, 1, :, :], bp2[:, :, 1, :, :]

                X = _features_s.permute([0, 1, 3, 4, 2]).flatten(0, 3).clone().cpu()
                y = _ds_gt_s.flatten().clone().cpu()

                valid_piexles = (_ds_gt_s != 255).flatten().cpu()
                X, y = X[valid_piexles], y[valid_piexles]
                X, y = X.numpy(), y.numpy()
                X = self.preproc(X)

                clf = exLogisticRegressionCV(n_splits=n_splits, Cs=Cs, sampling=self.sampling)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    clf.fit(X, y)

                X_te = features_s[:, idx_shot != i_shot, :, :, :].permute([0, 1, 3, 4, 2])
                X_te = X_te.flatten(0, 3).clone().cpu()
                y_te = ds_gt_s[:, idx_shot != i_shot, :, :].flatten().clone().cpu()

                valid_piexles = (ds_gt_s[:, idx_shot != i_shot, :, :] != 255)
                valid_piexles = valid_piexles.flatten().cpu()
                X_te, y_te = X_te[valid_piexles], y_te[valid_piexles]
                X_te, y_te = X_te.numpy(), y_te.numpy()

                num_classes = len(v)+1
                y_proba = clf.predict_proba_w_blank(X_te, num_classes)[0]
                y_pred = y_proba.argmax(axis=1)
                ap = []
                for yy in range(len(v)+1):
                    ap.append(average_precision_score(y_te == yy, y_proba[:, yy]))
                _score = np.mean(ap)
                val_scores[k][i_shot, :] = ap
                sk_model[k].append(clf)

            base_probas = self.base_classifier.get_base_probas(features_s)
            bpm2, bp2 = base_probas.topk(2, dim=2)
            base_probas_max, base_pred = bpm2[:, :, 0, :, :], bp2[:, :, 0, :, :]
            base_probas_max2, base_pred2 = bpm2[:, :, 1, :, :], bp2[:, :, 1, :, :]

            X = features_s.permute([0, 1, 3, 4, 2]).flatten(0, 3).clone().cpu()
            y = ds_gt_s.flatten().clone().cpu()

            valid_piexles = (ds_gt_s != 255).flatten().cpu()
            X, y = X[valid_piexles], y[valid_piexles]
            X, y = X.numpy(), y.numpy()
            X = self.preproc(X)

            clf = exLogisticRegressionCV(n_splits=n_splits, Cs=Cs, sampling=self.sampling)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                clf.fit(X, y)
            sk_model[k].append(clf)

        self.sk_model = sk_model


    @torch.no_grad()
    def predict(self, features_q, H, W):
        device = features_q.get_device()

        base_probas = self.base_classifier.get_base_probas(features_q)
        n_task, shots, num_bg_base_classes, h, w = base_probas.size()

        if (h, w) != (H, W):
            base_probas = F.interpolate(base_probas.view(n_task * shots, num_bg_base_classes, h, w),
                    size=(H, W), mode='bilinear', align_corners=True)
            base_probas = base_probas.view(n_task, shots, num_bg_base_classes, H, W)
        base_predict = base_probas.argmax(dim=2)

        final_predict = base_predict.clone()

        X = features_q.permute([0, 1, 3, 4, 2]).flatten(0, 3).clone().cpu().numpy()
        X = self.preproc(X)

        for k, v in self.table.items():
            target_base_flag = base_predict == k
            if target_base_flag.sum() == 0:
                continue

            num_classes = len(v) + 1
            if self.shot == 1 or not self.ensemble:
                bg_novel_probas = self.sk_model[k][0].predict_proba(X)
            else:
                ww = np.array([1, 1, 1, 1, 1, 5])
                bg_novel_probas, flags = [], []
                for i_shot in range(self.shot+1):
                    proba, flag = self.sk_model[k][i_shot].predict_proba_w_blank(X, num_classes)
                    bg_novel_probas.append(ww[i_shot]*proba)
                    flags.append(ww[i_shot]*flag)
                bg_novel_probas = np.stack(bg_novel_probas).sum(axis=0)
                flags = np.stack(flags)
                bg_novel_probas /= flags.sum(axis=0, keepdims=True)

            bg_novel_probas = bg_novel_probas.reshape((n_task*shots, h, w, num_classes))
            bg_novel_probas = torch.from_numpy(bg_novel_probas)
            bg_novel_probas = torch.permute(bg_novel_probas, (0, 3, 1, 2)).to(device)

            if (h, w) != (H, W):
                bg_novel_probas = F.interpolate(bg_novel_probas.view(n_task * shots, num_classes, h, w),
                    size=(H, W), mode='bilinear', align_corners=True).view(
                    n_task, shots, num_classes, H, W)
            bg_novel_pred = bg_novel_probas.argmax(dim=2)
            override = target_base_flag & (bg_novel_pred != 0)
            final_predict[override] = v.to(device)[bg_novel_pred[override] - 1]

        return final_predict


def proba_wo_d4p(clf, X_te):
    if isinstance(clf, RandomForestClassifier):
        decision = clf.predict_proba(X_te)
    else:
        decision = clf.decision_function(X_te)
    if decision.ndim == 1:
        decision_2d = np.c_[-decision, decision]
    else:
        decision_2d = decision
    return softmax(decision_2d, copy=False)


def under_sampling(X, y):
    classes, counts = np.unique(y, return_counts=True)
    n_min = counts[classes != 0].min()

    idx = []
    for c in classes:
        c_idx = np.where(y == c)[0]
        s = np.random.permutation(len(c_idx))[:n_min]
        c_idx = c_idx[s]
        idx.append(c_idx)
    idx = np.concatenate(idx, axis=0)
    return idx


def over_sampling(X, y):
    classes, counts = np.unique(y, return_counts=True)
    n_max = counts[classes != 0].max()

    idx = []
    for c in classes:
        c_idx = np.where(y == c)[0]
        if len(c_idx) < n_max:
            c_idx = np.random.choice(c_idx, size=n_max, replace=True)
        elif len(c_idx) > n_max:
            c_idx = np.random.choice(c_idx, size=n_max, replace=False)
        idx.append(c_idx)
    idx = np.concatenate(idx, axis=0)
    return idx


class exLogisticRegressionCV(BaseEstimator):

    def __init__(self, n_splits, Cs, sampling=None):
        self.n_splits = n_splits
        self.Cs = Cs
        self.sampling = sampling

    def fit(self, X, y):
        self.le = LabelEncoder().fit(y)
        self.classes_ = self.le.classes_
        y = self.le.transform(y)

        if type(self.Cs) is int:
            n_C = self.Cs
            Cs = np.logspace(-5, 5, n_C)
        elif type(self.Cs) in [list, np.array]:
            n_C = len(self.Cs)
        else:
            raise RuntimeError()

        skf = StratifiedKFold(n_splits=self.n_splits)
        scores = np.zeros((self.n_splits, n_C))
        for i, (train_index, test_index) in enumerate(skf.split(X, y)):
            X_tr, y_tr, X_te, y_te = X[train_index], y[train_index], X[test_index], y[test_index]
            X_tr, y_tr = self.handle_imbalance(X_tr, y_tr)

            for j, C in enumerate(Cs):
                clf = LogisticRegression(C=C)
                clf.fit(X_tr, y_tr)
                y_proba = proba_wo_d4p(clf, X_te)
                y_pred = y_proba.argmax(axis=1)
                ap = []
                for yy in np.unique(y_te):
                    ap.append(average_precision_score(y_te == yy, y_proba[:, yy]))
                _score = np.mean(ap)
                scores[i, j] = _score

        Cs_idx = np.argmax(scores.mean(axis=0))
        X, y = self.handle_imbalance(X, y)
        clf = LogisticRegression(C=Cs[Cs_idx])
        clf.fit(X, y)
        self.clf = clf
        return self

    def handle_imbalance(self, X, y):
        if self.sampling == 'bg':
            _, counts = np.unique(y, return_counts=True)
            max_num_novel = np.sort(counts)[-2]
            idxs = np.where(y == 0)[0]
            if len(idxs) > max_num_novel:
                idxs = np.random.choice(idxs, size=max_num_novel, replace=False)
            idxs = np.concatenate([idxs, np.where(y != 0)[0]])
            X, y = X[idxs], y[idxs]
        elif self.sampling == 'us':
            idxs = under_sampling(X, y)
            X, y = X[idxs], y[idxs]
        elif self.sampling == 'os':
            idxs = over_sampling(X, y)
            X, y = X[idxs], y[idxs]
        else:
            raise RuntimeError()
        return X, y

    def decision_function(self, X):
        if isinstance(self.clf, RandomForestClassifier):
            return self.clf.predict_proba(X)
        return self.clf.decision_function(X)

    def predict_proba(self, X):
        return proba_wo_d4p(self.clf, X)

    def predict_proba_w_blank(self, X, n_classes):
        # to avoid error when support set does not contain all novel classes
        proba = np.zeros((X.shape[0], n_classes))
        p = proba_wo_d4p(self.clf, X)
        proba[:, self.le.classes_] = p
        flag = np.zeros(n_classes)
        flag[self.le.classes_] = 1
        return proba, flag

    def predict(self, X):
        return self.le.inverse_transform(self.predict_proba(X).argmax(axis=1))
