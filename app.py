from flask import Flask, render_template, request, url_for, session, redirect, flash
from pymongo import MongoClient 
from functools import wraps
import datapreprocessing
import new_classification
from keylogger_process import Process_thread
import time
import csv
username = None
app = Flask(__name__)

client = MongoClient('localhost:27017',retryWrites=False)
db = client.jenil
collections = db['User']

datapreprocessing.main()
clf = datapreprocessing.get_clf()
breakout = 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/register', methods = ['GET','POST'])
def register():
    if request.method == 'POST':
        a=request.form['username']
        b=request.form['password']
        c=request.form['gen']
        d=request.form['contact']
        e=request.form['age']
        collections.insert({"username": a, "password": b, "gen": c, "contact": d ,"age":e})
        flash('You are register and can now login in','success')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods = ['GET','POST'])
def login():
    if request.method == 'POST':
        res = collections.find({})

        global username
        username = request.form['username']
        password = request.form['password']

        for doc in res:
            if doc['username'] == username and doc['password'] == password:
                doc.pop('password', None)
                doc.pop('_id', None)
                session['logged_in'] = True
                session['username'] = username
                flash('You are now logged in','success')
                try:
                    doc.pop('result', None)
                except KeyError:
                    pass
                return render_template("profile.html", user=username, result=doc)
    
    return render_template('login.html')

def is_logged_in(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return f(*args, **kwargs)
        else:
            flash('Unauthorized, Please login', 'danger')
            return redirect(url_for('login'))
    return wrap

@app.route('/profile', methods = ['GET','POST'])
@is_logged_in
def profile():
    global username
    p = Process_thread(username)
    p.start()
    global breakout
   # p.join()
    while True:

        with open("./user_csv/" + username + ".csv", 'r') as f:
            reader = csv.reader("./user_csv/" + username + ".csv")
            row_count = len(list(reader))
            row_count = sum(1 for row in f)

            if row_count > 20:
                prediction = get_prediction("./user_csv/" + username + ".csv")
                print(db.User.update({"username": username},
                                 {"$set": {"Parkinsons": str(prediction)}}))
                res = db.User.find({})

                for doc in res:
                    if doc['username'] == username:
                        breakout = 1
                        doc.pop('password', None)
                        doc.pop('_id', None)
                        try:
                            doc.pop('result', None)
                        except KeyError:
                            pass
                        time.sleep(60)
                        
                        return render_template("profile.html", user=username, result=doc)
        if breakout == 1:
            p.stop()
            break
        time.sleep(60)
    return


@app.route('/logout')
@is_logged_in
def logout():
    session.clear()
    flash('You are now logged out', 'success')
    return redirect(url_for('login'))

def get_prediction(file_name):
    global clf
    test = new_classification.generate_typing_df(file_name)
    print('Your probability of parkinsons' + str(clf.predict(test)))
    if clf.predict(test)[0] == 0:
        return False
    else:
        return True

if __name__ == '__main__':
    app.secret_key = 'secret123'
    app.run(debug=True)