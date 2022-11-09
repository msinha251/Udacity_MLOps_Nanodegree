from flask import Flask, request

app = Flask(__name__)

@app.route('/')
def index():
    user = request.args.get('user')
    return "Hello, {}".format(user)

if __name__ == '__main__':
    app.run(debug=True)