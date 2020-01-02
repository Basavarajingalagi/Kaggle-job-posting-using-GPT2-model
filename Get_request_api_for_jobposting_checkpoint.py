import gpt_2_simple as gpt2
from flask import Flask,request,jsonify
import tensorflow as tf


app = Flask(__name__)


@app.route('/jobtitle=<string:title>&count=<int:num>',methods = ['GET'])
def output_jsson(title,num):
    sess = gpt2.start_tf_sess()
    gpt2.load_gpt2(sess, run_name='run1')

    

    text = gpt2.generate(sess,
                         length=250,
                         temperature=0.7,
                         prefix=title,
                         nsamples=num,
                         batch_size=num,
                         return_as_list=True
                         )
    suggetion_val= []
    for i in text:
        
        suggetion_val.append({'key':i})


    return jsonify({'title': title,
                    'suggestion': suggetion_val})

if __name__ == '__main__':
    app.run(debug=True)
