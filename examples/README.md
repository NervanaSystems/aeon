# Examples

Examples present basic aeon usage scenarios.

## distributed aeon
To run `cpp_remote_iterator`, `python_remote_iterator` and `python_remote_iterator_shared`, it's required to run `aeon-service` from `service` directory with URI provided. Sample usage:


    # run aeon-service
    ./aeon-service --uri "http://127.0.0.1:4586"
    # run remote iterator example - manifest file should be visible to aeon-service
    ./remote_iterator.py -a 127.0.0.1 -p 4586 -m ../../../test/test_data/manifest.tsv
