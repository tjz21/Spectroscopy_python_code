
## Running tests
run all tests with 
`python -m unittest -v`

to run a single test, for example the MD tests,
python -m unittest -v tests.test_MD

## Coverage Reports
So see the code coverage by the tests, install the `coverage` package with pip:
`python3 -m pip install coverage`

Then, first run a discoveragr test run:
`coverage run -m unittest discover`

You can then view a coverage report in the terminal,
`coverage report`

or with a nicer html view,
`coverage html`

See the Coverage docs for more information: https://coverage.readthedocs.io/en/7.2.7/
