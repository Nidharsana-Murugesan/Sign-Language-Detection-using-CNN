from flask import Flask, render_template, redirect, url_for
import subprocess

app = Flask(__name__)

@app.route('/')
def homepage():
    return render_template('index.html')

@app.route('/run_camera')
def run_camera():
    # Run the Python script for hand gesture recognition
    subprocess.Popen(["python", "demo.py"])
    return render_template('run_camera.html')

if __name__ == '__main__':
    app.run(debug=True)
