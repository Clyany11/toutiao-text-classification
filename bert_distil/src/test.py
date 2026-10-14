# coding: UTF-8
import numpy as np
import torch
import time
from test_eval import test
from importlib import import_module
import argparse
from utils import (
    build_dataset,
    build_iterator,
    get_time_dif,
    build_dataset_CNN,
)

parser = argparse.ArgumentParser(description="Chinese Text Classification")
parser.add_argument(
    "--task",
    type=str,
    required=True,
    help="choose a task: test_bert, or test_kd",
)
args = parser.parse_args()

if __name__ == "__main__":
    dataset = "toutiao"
    if args.task == "test_bert":
        model_name = "bert"
        x = import_module("models." + model_name)
        config = x.Config(dataset)
        np.random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.deterministic = True  # 保证每次结果一样
        print("Loading data for Bert Model...")
        train_data, dev_data, test_data = build_dataset(config)
        test_iter = build_iterator(test_data, config)
        model = x.Model(config).to(config.device)
        test(config, model, test_iter)

    if args.task == "test_kd":
        model_name = "textCNN"
        cnn_module = import_module("models." + model_name)
        cnn_config = cnn_module.Config(dataset)
        np.random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.deterministic = True  # 保证每次结果一样
        # 构建cnn数据集
        vocab, cnn_train_data, cnn_dev_data, cnn_test_data = build_dataset_CNN(
            cnn_config
        )
        cnn_test_iter = build_iterator(cnn_test_data, cnn_config)
        cnn_config.n_vocab = len(vocab)
        # 加载student模型
        cnn_model = cnn_module.Model(cnn_config).to(cnn_config.device)
        test(cnn_config, cnn_model, cnn_test_iter)
