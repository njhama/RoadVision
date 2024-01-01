from flask import Flask, request, jsonify
import threading
import sys
import os


sys.path.append('../')
import gen

app = Flask(__name__)

@app.route('/train', methods=['POST'])
def train():
    data = request.json
    
    threading.Thread(target=gen.train_gan, args=(data,)).start()
    return jsonify({"message": "Training started"}), 200

if __name__ == '__main__':
    app.run(debug=True)
