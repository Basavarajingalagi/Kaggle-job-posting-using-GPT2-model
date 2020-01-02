import gpt_2_simple as gpt2
from flask import Flask,request,jsonify
import tensorflow as tf


app = Flask(__name__)
#jobtitle=<string:title>&count=<int:num>

@app.route('/',methods = ['POST'])
def output_jsson():

    sess = gpt2.start_tf_sess()
    gpt2.load_gpt2(sess, run_name='run1')

    

    text = gpt2.generate(sess,
                         length=250,
                         temperature=0.7,
                         prefix=request.json["title"],
                         nsamples=int(request.json["num"]),
                         batch_size=int(request.json["num"]),
                         return_as_list=True
                         )
    suggetion_val= []
    for i in text:
        suggetion_val.append({'key':i})


    return jsonify({'title': request.json["title"],
                    'suggestion': suggetion_val})

if __name__ == '__main__':
    app.run(debug=True)
