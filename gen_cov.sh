#!/bin/sh
lcov --base-directory . --directory . --gcov-tool ../llvm-gcov.sh --capture -o cov.info
lcov --remove ./cov.info '/opt/*' '/usr/*' 'gtest/*' -o coverage.info
genhtml coverage.info -o html-coverage-raport 
