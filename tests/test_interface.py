from objects.interface import Interface
from objects.parameters import CUBE_SIDE_LENGTH


def test_make_interface():

    interface = Interface(label="test")


if __name__ == "__main__":
    import pytest

    pytest.main(["tests"])
