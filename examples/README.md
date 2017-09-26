# Examples

Examples present basic aeon usage scenarios.

## distributed aeon
To run `cpp_remote_iterator` and `python_remote_iterator`, it's required to run `aeon-server` from `server` directory with address and port provided. Sample usage:


    cd server
    ./aeon-server --a http://127.0.0.1 -p 34568 &
    cd ../python_remote_iterator/
    ./remote_iterator.py -a 127.0.0.1 -p 34568
