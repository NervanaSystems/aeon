# Examples

Examples present basic aeon usage scenarios.

## distributed aeon
To run `cpp_remote_iterator`, `python_remote_iterator` and `python_remote_iterator_shared`, it's required to run `aeon-service` from `service` directory with URI provided. Sample usage:


    cd service
    ./aeon-service --uri "http://127.0.0.1:80"
    cd ../python_remote_iterator/
    ../../../test/test_data/manifest.tsv
