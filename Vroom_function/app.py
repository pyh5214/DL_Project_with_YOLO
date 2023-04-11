import os, sys
from flask import Flask, escape, request,  Response, g, make_response, flash
from flask.templating import render_template
from neural_style_transfer import *

 
 

app = Flask(__name__)
app.debug = True


 
@app.route('/')
def nst_get():
    return render_template('nst_get.html')
 
@app.route('/nst_post', methods=['GET','POST'])
def nst_post():
    if request.method == 'POST':
 
        # User Image (target image)
        user_img = request.files['user_img']
        user_img.save('./static/images/'+str(user_img.filename))
        user_img_path = '/images/'+str(user_img.filename)
 
        # Neural Style Transfer 
        # transfer_img = main(refer_img_path, user_img_path)
        # transfer_img_path = '../static/images/nst_get/'+str(transfer_img.split('/')[-1])
        test = main(user_img_path)

        
        if test == 0:
            return render_template('nst_post.html', 
                    user_img=user_img_path, test=test)
        
        else :
            return render_template('retry.html', user_img = user_img_path) 
        
        
@app.route('/loading')
def loading():
    return render_template('loading.html')
        
 
    
  
  
if __name__ == "__main__":
    app.run(host = '0.0.0.0')