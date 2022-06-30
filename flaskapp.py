#  http://127.0.0.1:80/ 
#  set FLASK_APP=flaskapp
#  flask run
from base64 import decode
from typing_extensions import Self
from flask import Flask, render_template, request
import flask_wtf 
import string
from flask_wtf import Form
from wtforms import SelectField, SubmitField
import deocder_plot

class configform(Form):
    dataset = SelectField('数据集', choices=[('MNIST','MNIST'),('FashionMNIST','FashionMNIST'),('AdvMNIST','AdvMNIST')])
    method = SelectField('度量方法', choices=[('Ensemble','Ensemble'),('MCDropout','MCDropout')])
    index = SelectField('度量指标', choices=[('Entrropy','Entropy'),('MI','MI')])
    submit = SubmitField("确定")

class configtext():
    def __init__(self):
        self.dataset=""
        self.net=""
        self.method=""
        self.index=""

class caculate():
    def __init__(self):
        self.pre_class=0
        self.pre_prob=0
        self.MI=0
        self.Ent=0

class point():
    def __init__(self):
        self.x=0
        self.y=0

app = Flask(__name__)
app.secret_key = 'development key'
@app.route("/",methods = ['GET', 'POST'])
def hello_world():
    form = configform()
    img_url=""
    text = configtext()
    cal = caculate()
    dot = point()
    text.dataset = 'MNIST' 
    text.net = 'LeNet5'
    text.method = 'MC'
    text.index = 'Ent'
    dot.x = 0.0
    dot.y = 0.0
    # cal.Ent, cal.pre_class, cal.pre_prob, cal.MI= 0
    if request.method == 'POST':
        text.dataset = request.values.get("dataset") 
        text.net = request.values.get("net")
        text.method = request.values.get("method")
        text.index = request.values.get("index")
        dot.x = float(request.values.get("X"))
        dot.y = float(request.values.get("Y"))
        cal.pre_class, cal.pre_prob, cal.MI, cal.Ent = deocder_plot.get_lantent_picture(dot.x,
                                                                                        dot.y, 
                                                                                        text.dataset,
                                                                                        text.method,
                                                                                        text.net)
        img_url= '../static/Result_picture/'+text.dataset+'_'+text.net+'_'+text.method+'_'+text.index+'.png'
        kwargs = {
            "img_url": img_url,
            "text": text,
            "cal": cal,
            "dot": dot
        }
        return render_template('app.html', **kwargs) 
    else:
        cal.pre_class, cal.pre_prob, cal.MI, cal.Ent = deocder_plot.get_lantent_picture(dot.x,
                                                                                        dot.y, 
                                                                                        text.dataset,
                                                                                        text.method,
                                                                                        text.net)
        img_url= '../static/Result_picture/'+text.dataset+'_'+text.net+'_'+text.method+'_'+text.index+'.png'
        kwargs = {
            "img_url": img_url,
            "text": text,
            "cal": cal,
            "dot": dot
        }
        return render_template('app.html', **kwargs) 
    #return render_template('app.html', form = form) 

if __name__ == '__main__':
    app.run(host='127.0.0.1',debug=True,port=80) #127.0.0.1 回路 自己返回自己