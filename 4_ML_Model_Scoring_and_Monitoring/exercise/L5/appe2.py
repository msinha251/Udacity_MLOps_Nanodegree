
from flask import Flask, request
import pandas as pd


def readpandas(filename):
    df = pd.read_csv(filename)
    return df

app = Flask(__name__)

@app.route('/')
def index():
    user = request.args.get('user')
    return "Hello " + user + '\n'

@app.route('/size')
def size():
    filename = request.args.get('filename')
    df = readpandas(filename)
    return str(len(df.index))


@app.route('/summary')
def summary():
    filename = request.args.get('filename')
    df = readpandas(filename)
    return str(df.mean(axis=0))
if __name__ == '__main__':
    app.run(debug=True)




