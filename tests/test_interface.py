from objects.interface import Interface
from objects.parameters import INTERFACE_PATH


def test_construct_interface():
    interface = Interface(INTERFACE_PATH, label="Z_456")
    interface.mesh.show()


if __name__ == "__main__":
    import pytest

    pytest.main(["tests/test_interface.py"])
