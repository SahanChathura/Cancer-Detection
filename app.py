from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify
from flask_sqlalchemy import SQLAlchemy
import os.path
import os
import shutil
import random
import torch
import torchvision
import numpy as np

from PIL import Image
from matplotlib import pyplot as plt

torch.manual_seed(0)

app = Flask(__name__)
app.secret_key = "Secret Key"
wsgi_app = app.wsgi_app

# SqlAlchemy Database Configuration With Mysql
DATA = 'mysql+mysqlconnector://{user}:{password}@{server}/{database}'.format(user='root',
                                                                             password='root',
                                                                             server='localhost',
                                                                             database='cancerpro')
app.config['SQLALCHEMY_DATABASE_URI'] = DATA
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)


class Cancertypes(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    did = db.Column(db.String(100))
    dname = db.Column(db.String(100))
    phone = db.Column(db.String(100))
    img = db.Column(db.String(200))
    mresult = db.Column(db.String(100))

    def __init__(self, did, dname, phone, img, mresult):
        self.did = did
        self.dname = dname
        self.phone = phone
        self.img = img
        self.mresult = mresult


def load_checkpoint(model, optimizer, load_path):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

    return model, optimizer, epoch


@app.route('/login')
def Index():
    all_data = Cancertypes.query.all()
    return render_template("index.html", employees=all_data)


@app.route('/')
def login():
    return render_template("login.html")


@app.route('/checklogin', methods=['POST'])
def checklogin():
    if request.method == 'POST':
        userID = request.form['userid']
        userPass = request.form['password']

        if userID == "root" and userPass == "root@123":
            all_data = Cancertypes.query.all()
            return render_template("index.html", employees=all_data)
        return render_template("login.html")


@app.route('/insert', methods=['POST'])
def insert():
    if request.method == 'POST':
        did = request.form['did']
        dname = request.form['dname']
        phone = request.form['phone']

        file = request.files['file']
        pathname = os.path.join("uploads", file.filename)
        file.save(pathname)

        # machine learning model sarting
        resnet18 = torchvision.models.resnet18(pretrained=True)

        resnet18.fc = torch.nn.Linear(in_features=512, out_features=3)
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(resnet18.parameters(), lr=3e-5)

        model, optimizer, epoch = load_checkpoint(resnet18, optimizer, 'new3.model')

        test_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=(224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        image = Image.open(pathname).convert('RGB')
        im = test_transform(image).reshape(1, 3, 224, 224)
        im.shape
        output = model(im)
        _, preds = torch.max(output, 1)
        print(int(preds[0]))
        la = ["normal", "adenocarcinoma", "squamouscellcarcinoma"]
        predictions = la[int(preds[0])]

        my_data = Cancertypes(did, dname, phone, file.filename, predictions)
        db.session.add(my_data)
        db.session.commit()

        flash("DATA ADDED")

        return redirect(url_for('Index'))


@app.route('/update', methods=['GET', 'POST'])
def update():
    if request.method == 'POST':
        my_data = Cancertypes.query.get(request.form.get('id'))

        my_data.did = request.form['did']
        my_data.dname = request.form['dname']
        my_data.phone = request.form['phone']

        file = request.files['file']
        pathname = os.path.join("uploads", file.filename)
        file.save(pathname)
        my_data.img = pathname

        db.session.commit()
        flash("Employee Updated Successfully")

        return redirect(url_for('Index'))


@app.route('/delete/<id>/', methods=['GET', 'POST'])
def delete(id):
    my_data = Cancertypes.query.get(id)
    db.session.delete(my_data)
    db.session.commit()
    flash("Employee Deleted Successfully")

    return redirect(url_for('Index'))


@app.route('/show/<name>/', methods=['GET', 'POST'])
def show(name):
    return send_from_directory("uploads", name)


@app.route('/mobileapp/alldata')
def mobilealldata():
    all_data = Cancertypes.query.all()
    details = []
    content = {}
    for a in all_data:
        content = {'id': a.id, 'did': a.did, 'name': a.dname, 'result': a.mresult}
        details.append(content)
        content = {}

    return jsonify(details)


if __name__ == "__main__":
    app.run(debug=True)
