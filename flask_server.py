from keras.models import model_from_json
from konlpy.tag import Okt
from keras_preprocessing.text import Tokenizer
import numpy as np
from keras.preprocessing import sequence
import pickle

class Model():

    def __init__(self):
        # 모델 로드
        json_file = open("model.json", "r")
        loaded_model_json = json_file.read()
        json_file.close()
        self.loaded_model = model_from_json(loaded_model_json)

        # 모델 가중치 로드
        self.loaded_model.load_weights("model.h5")

        # tokenizer 로드
        with open('tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)

        self.okt = Okt()
        self.max_len = 75


    def prediction(self, text):
        t = self.okt.pos(text, stem=True, norm=True)
        s = self.tokenizer.texts_to_sequences([[i[0] + '/' + i[1]] for i in np.array(t)])

        #   0 으로 채우기
        #     sn = [n if n!=[] else [0] for n in s]

        #   0 제거
        sn = []
        for n in s:
            if n != []:
                sn.append(n)

        print(text)
        print(sn)

        p = sequence.pad_sequences([sn], maxlen=self.max_len)
        x = np.array(p).reshape((1, -1))
        y = self.loaded_model.predict(x)
        return y[0, 0], sn


    def processing(self, text='.'):
        y, sn = self.prediction(text)
        print('=================================================')
        print(sn)
        if y < 0.5:
            print('"%s"는\n%f의 확률로 부정적입니다.' % (text, (1 - y) * 100))
        else:
            print('"%s"는\n%f의 확률로 긍정적입니다.' % (text, y * 100))
        print('=================================================')



# processing('아 이거 보면서 너무 화나네요 뭣하러 이런거 찍어내는지 에휴..')

from flask import Flask, request

app = Flask(__name__)

@app.route("/")
def root():
    return '''
        <h2>텍스트 긍부정 분류 모델</h2>
        <form action="query" accept-charset="utf-8" >
            <textarea name="content" placeholder="텍스트 입력" rows="5" cols="100"></textarea>
            <input type="submit" value="실행"/>
        </form>
    '''

@app.route("/query")
def query():
    content = request.args.get('content')
    y, sn = model.prediction(content)

    form = '''
        <h2>텍스트 긍부정 분류 모델</h2>
        <form action="query" accept-charset="utf-8" >
            <textarea name="content" placeholder="텍스트 입력" rows="5" cols="100"></textarea>
            <input type="submit" value="실행"/>
        </form>
    '''

    if y < 0.5:
        return form+'<strong>"%s"</strong>는\n%f의 확률로 <strong>부정적</strong>입니다. <br> <textarea rows="5" cols="100">%s</textarea>' % (content, (1 - y) * 100, str(sn))
    else:
        return form+'<strong>"%s"</strong>는\n%f의 확률로 <strong>긍정적</strong>입니다. <br> <textarea rows="5" cols="100">*log\n%s</textarea>' % (content, y * 100, str(sn))

host = '0.0.0.0'
post = '8000'

if __name__ == '__main__':
    model = Model()
    # 서버와 같이 쓰기위해 동기 처리
    model.loaded_model._make_predict_function()
    app.run(host=host, port=post)