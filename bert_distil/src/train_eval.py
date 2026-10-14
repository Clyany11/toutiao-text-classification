# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from transformers.optimization import AdamW
from tqdm import tqdm
import math
import logging


def fetch_teacher_outputs(teacher_model, train_iter):
    teacher_model.eval()
    teacher_outputs = []
    with torch.no_grad():
        for i, (data_batch, labels_batch) in enumerate(train_iter):
            outputs = teacher_model(data_batch)
            teacher_outputs.append(outputs)
    return teacher_outputs


criterion = nn.KLDivLoss()


def loss_fn(outputs, labels):
    # 一定要注意先定义类对象,再输入函数参数,不能写成nn.CrossEntropyLoss(outputs,labels),少了类对象的括号()
    return nn.CrossEntropyLoss()(outputs, labels)


# 编写实现KL散度的损失函数的代码
def loss_fn_kd(outputs, labels, teacher_outputs):
    # 注意:pytorch中的KL散度nn.KLDivLoss要求student输入为log-probabilities,软目标为probabilities
    # 关于API函数nn.KLDivLoss(), 第1个参数必须是经历log计算后的分布值, 第2个参数必须是没有log计算的分布值
    alpha = 0.8
    T = 2
    # 软目标损失
    # 首先计算学生网络的带有T参数的log_softmax输出分布
    output_student = F.log_softmax(outputs / T, dim=1)

    # 然后计算教师网络的带有T参数的softmax输出分布
    output_teacher = F.softmax(teacher_outputs / T, dim=1)

    # 计算软目标损失,使用KLDivLoss(),第一个参数为student网络输出, 第二个参数为teacher网络输出
    soft_loss = criterion(output_student, output_teacher)
    # 硬目标损失
    # 即学生网络的输出概率和真实标签之间的损失, 因为真实标签是one-hot编码, 因此直接使用交叉熵损失即可
    hard_loss = F.cross_entropy(outputs, labels)

    # 计算总损失
    # 原始论文中已经证明, 引入T会导致软目标产生的梯度和真实目标产生的梯度相比只有1/(T*T)
    # 因此计算完软目标的loss值后要乘以T^2.
    KD_loss = soft_loss * alpha * T * T + hard_loss * (1.0 - alpha)

    return KD_loss


def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train()
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float("inf")
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    model.train()
    for epoch in range(config.num_epochs):
        print("Epoch [{}/{}]".format(epoch + 1, config.num_epochs))
        for i, (trains, labels) in enumerate(tqdm(train_iter)):
            model.zero_grad()
            outputs = model(trains)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = "*"
                    last_improve = total_batch
                else:
                    improve = ""
                time_dif = get_time_dif(start_time)
                msg = "Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}"
                print(
                    msg.format(
                        total_batch,
                        loss.item(),
                        train_acc,
                        dev_loss,
                        dev_acc,
                        time_dif,
                        improve,
                    )
                )
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    test(config, model, test_iter)


def train_kd(
    bert_config,
    cnn_config,
    bert_model,
    cnn_model,
    bert_train_iter,
    cnn_train_iter,
    cnn_dev_iter,
    cnn_test_iter,
):
    start_time = time.time()
    param_optimizer = list(cnn_model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=cnn_config.learning_rate)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float("inf")
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    cnn_model.train()
    loading_start = time.time()
    bert_model.eval()
    teacher_outputs = fetch_teacher_outputs(bert_model, bert_train_iter)
    elapsed_time = math.ceil(time.time() - loading_start)
    logging.info(
        "- Finished computing teacher outputs after {} secs..".format(elapsed_time)
    )
    for epoch in range(cnn_config.num_epochs):
        print("Epoch [{}/{}]".format(epoch + 1, cnn_config.num_epochs))
        for i, (trains, labels) in enumerate(tqdm(cnn_train_iter)):
            cnn_model.zero_grad()
            outputs = cnn_model(trains)
            loss = loss_fn_kd(outputs, labels, teacher_outputs[i])
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(cnn_config, cnn_model, cnn_dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(cnn_model.state_dict(), cnn_config.save_path)
                    improve = "*"
                    last_improve = total_batch
                else:
                    improve = ""
                time_dif = get_time_dif(start_time)
                msg = "Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}"
                print(
                    msg.format(
                        total_batch,
                        loss.item(),
                        train_acc,
                        dev_loss,
                        dev_acc,
                        time_dif,
                        improve,
                    )
                )
                cnn_model.train()
            total_batch += 1
            if total_batch - last_improve > cnn_config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    test(cnn_config, cnn_model, cnn_test_iter)


def test(config, model, test_iter):
    # test
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
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
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
