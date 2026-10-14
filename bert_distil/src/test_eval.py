# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
import math
from utils import get_time_dif


def test(config, model, test_iter):
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(
        config, model, test_iter, test=True
    )
    msg = "Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}"
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    average_time = 0.0
    with torch.no_grad():
        for texts, labels in data_iter:
            start_time = time.time()
            outputs = model(texts)
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            end_time = time.time() - start_time
            average_time += end_time
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
    average = average_time / 10000
    print("Average predict time is:", average * 1000, "ms")
    print("*********************************")
    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(
            labels_all,
            predict_all,
            target_names=config.class_list,
            digits=4,
        )
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)
