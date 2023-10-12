
## Running tests
run all tests with 
`python -m unittest -v`

To run a specific test or set of tests, call the same command with `tests.file.class.function`. For example, 
`python -m unittest -v tests.test_MD.MD_cumulant.runTest`
will run the function `runTest` in the class `MD_cumulant` within the file `test_MD.py`, while
`python -m unittest -v tests.test_MD.MD_cumulant`
will run all tests in the class `MD_cumulant` within the file `test_MD.py`.

## Coverage Reports
So see the code coverage by the tests, install the `coverage` package with pip:
`python3 -m pip install coverage`

Then, first run a discoveragr test run:
`coverage run -m unittest discover`
Make sure to set `NUMBA_DISABLE_JIT=1` so that jit functions are counted
towards the total coverage.

You can then view a coverage report in the terminal,
`coverage report`

or with a nicer html view,
`coverage html`

See the Coverage docs for more information: https://coverage.readthedocs.io/en/7.2.7/
