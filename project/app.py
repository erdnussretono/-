from flask import Flask, render_template, request, jsonify, make_response, json
from transformer.utils.hparams import *
from transformer.models import transformer
from transformer.modules import predict
from cnn.cnn import padded_sequence
from util.findfacility import findfacility





app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


new_model = transformer(
    vocab_size=VOCAB_SIZE,
    num_layers=NUM_LAYERS,
    dff=DFF,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dropout=DROPOUT)
new_model.load_weights('transformer_M')
emotion_dict ={'pleasure' : 0 , 'panic':0, 'angry':0, 'unrest':0, 'wound':0, 'sad' : 0}
EMOTIONAL_PEAK = 4

# {0: '기쁨', 1: '당황', 2: '분노', 3: '불안', 4: '상처', 5: '슬픔'}


@app.route('/')
def main():
    
    resp = make_response(render_template('home.html'))
    for i in emotion_dict:
        print(i)
        resp.set_cookie(i, str(emotion_dict[i]))
    return resp


@app.route('/sos', methods=['POST'])
def sos():
    ipdata = request.remote_addr
    facility_list, lat, lon = findfacility(ipdata)
    return render_template('sos.html', facility = facility_list, lat = lat, lon = lon);

@app.route('/api', methods=['POST'])
def api():
    data = request.get_json()#json 데이터를 받아옴
    emotion = padded_sequence(data['sentence'])
    emotion_cnt = request.cookies.get(emotion)
    print(emotion_cnt)
    data['emotion'] = emotion

    data['emotion_cnt'] = 1
    data['pred'] = predict(data['sentence'], model=new_model)   
    
    return jsonify(data)# 받아온 데이터를 다시 전송
    

    




if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000) #,debug=True)
    