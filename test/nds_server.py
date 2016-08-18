import json

from flask import Flask, send_file
app = Flask(__name__)


@app.route("/object_count/")
def object_count():
    return json.dumps({
        'record_count': 200,
        'macro_batch_per_shard': 5,
    })


@app.route("/macrobatch/")
def macrobatch():
    return send_file('test.cpio')
    # return open('hello.cpio', 'rb').read()
    # return send_file('/usr/local/data/wdc/data/all-ingested/archive-0.cpio')


@app.route("/test_pattern/")
def test_pattern():
    return '0123456789abcdef' * 1024


if __name__ == "__main__":
    app.run()
