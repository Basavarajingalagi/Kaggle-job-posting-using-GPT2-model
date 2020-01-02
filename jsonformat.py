import gpt_2_simple as gpt2
from flask import Flask,request,jsonify
import tensorflow as tf


app = Flask(__name__)



@app.route('/')
def output_jsson():
    sess = gpt2.start_tf_sess()
    gpt2.load_gpt2(sess, run_name='run1')

    title = input('enter the title')

    text = gpt2.generate(sess,
                         length=250,
                         temperature=0.7,
                         prefix=title,
                         nsamples=5,
                         batch_size=5,
                         return_as_list=True
                         )
    suggetion_val= []
    for i in text:
        
        suggetion_val.append({'key':i})


    return jsonify({
                    'suggestion': suggetion_val,
                    'title': title
                    })

if __name__ == '__main__':
    app.run(debug=True)
