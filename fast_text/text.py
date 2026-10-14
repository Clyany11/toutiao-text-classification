import requests
import time
# 定义请求url和传⼊的data
url = "http://0.0.0.0:5000/v1/main_server/"
data = {"uid": "AI-3-20260411", "text": "小米市值跌了十个点"}
start_time = time.time()
# 向服务发送post请求
res = requests.post(url, data=data)
cost_time = time.time() - start_time
# 打印返回的结果
print('输⼊⽂本:', data['text'])
print('分类结果:', res.text)
print('单条样本预测耗时: ', cost_time * 1000, 'ms')