import shutil
from distutils.dir_util import copy_tree

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--network-tests",
        action="store_true",
        default=False,
        help="Run tests that involve network operations. This increases test time.",
    )


def pytest_collection_modifyitems(config, items):
    run_network = config.getoption("--network-tests")
    skip_network = pytest.mark.skip(reason="need --network-tests option to run")
    for item in items:
        # All tests marked with `pytest.mark.network` get skipped unless
        # `--network-tests` passed
        if not run_network and ("network" in item.keywords):
            item.add_marker(skip_network)


@pytest.fixture(scope="session")
def save_path(tmpdir_factory):
    dir = tmpdir_factory.mktemp("temp_data", numbered=False)
    path = str(dir)
    copy_tree("tests/data", path)
    yield path + "/"
    shutil.rmtree(str(tmpdir_factory.getbasetemp()))
