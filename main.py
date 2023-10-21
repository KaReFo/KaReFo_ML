from flask import Flask
import ML as ml
import os
app = Flask(__name__)


@app.route('/karefoML/<name>')
def home(name):
    print(name)
   # ml.learningPhase()
    ml.isTrainedFlag = True
    output =ml.generatesingleoutput(name)
    return "generated output! %s" % output


if __name__ == "__main__":
    app.run(debug=True)