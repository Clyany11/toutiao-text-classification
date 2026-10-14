import fasttext
from pathlib import Path
# 服务框架使⽤Flask, 导⼊⼯具包
from flask import Flask
from flask import request
app = Flask(__name__)

model_save_path = "/mnt/workspace/AI/11投满分/toutiao_fasttext_1788884032.bin"
# 实例化fasttext对象, 并加载模型参数⽤于推断, 提供服务请求
model = fasttext.load_model(model_save_path)
print('FastText模型实例化完毕...')
# 设定投满分服务的路由和请求⽅法
@app.route('/v1/main_server/', methods=["POST"])
def main_server():
    # 接收来⾃请求方发送的服务字段
    uid = request.form['uid']
    text = request.form['text']
    # 模型是按字切分训练的, 这里对请求文本同样按字切分
    input_text = ' '.join(list(text))
    # 执⾏预测
    res = model.predict(input_text)
    predict_name = res[0][0]
    return predict_name