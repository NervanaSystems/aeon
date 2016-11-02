import json

from flask import Flask, send_file, request, abort
import logging

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)

def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()

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


@app.route('/shutdown/')
def shutdown():
    shutdown_server()
    return 'Server shutting down...'


@app.route('/error')
def error():
    abort(400)


def run_server():
    app.run()


def run_server_with_timeout(seconds):
    """ run server in a seperate process and terminate it after `seconds` """
    import time
    from multiprocessing import Process

    server = Process(target=run_server)
    server.start()

    server.join(seconds)

    if server.is_alive():
        print("nds_server still active, killing")
        server.terminate()
        server.join()
        print("nds_server gone")


if __name__ == "__main__":
    # only run for 15 seconds, then die
    run_server_with_timeout(15)
