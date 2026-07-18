import metaxuda

def test_version():
    assert metaxuda.__version__ == "2.0.1"

def test_public_api():
    for name in metaxuda.__all__:
        assert hasattr(metaxuda, name)