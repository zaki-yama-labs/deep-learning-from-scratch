from deep_learning_from_scratch.ch02.perceptron import AND, NAND, OR, XOR


def test_AND():
    assert AND(0, 0) == 0
    assert AND(1, 0) == 0
    assert AND(0, 1) == 0
    assert AND(1, 1) == 1


def test_NAND():
    assert NAND(0, 0) == 1
    assert NAND(1, 0) == 1
    assert NAND(0, 1) == 1
    assert NAND(1, 1) == 0


def test_OR():
    assert OR(0, 0) == 0
    assert OR(1, 0) == 1
    assert OR(0, 1) == 1
    assert OR(1, 1) == 1


def test_XOR():
    assert XOR(0, 0) == 0
    assert XOR(1, 0) == 1
    assert XOR(0, 1) == 1
    assert XOR(1, 1) == 0
