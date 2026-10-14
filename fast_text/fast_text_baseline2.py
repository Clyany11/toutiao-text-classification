import fasttext
import time
train_data_path = 'data/train_fast1.txt'
dev_data_path = 'data/dev_fast.txt'
test_data_path = 'data/test_fast1.txt'
# autotuneValidationFile参数需要指定验证数据集所在路径,
# 它将在验证集上使⽤随机搜索⽅法寻找可能最优的超参数.
# 使⽤autotuneDuration参数可以控制随机搜索的时间, 默认是300s,
# 根据不同的需求, 我们可以延⻓或缩短时间.
# verbose: 该参数决定⽇志打印级别, 当设置为3, 可以将当前正在尝试的超参数打印出来.
model = fasttext.train_supervised(input=train_data_path,
autotuneValidationFile=dev_data_path,
autotuneDuration=600,
wordNgrams=2,
verbose=3)
# 在测试集上评估模型的表现
result = model.test(test_data_path)
print(result)
# 模型保存
time1 = int(time.time())
model_save_path = "./toutiao_fasttext_{}.bin".format(time1)
model.save_model(model_save_path)