
# coding: utf-8

# Tests for encoding module
#
# @author Álvaro Barbero Jiménez

from neurowriter.encoding import Encoder


def test_reversible_encoding():
    """Encoding a text and decoding it produces the same result"""
    text = "For the glory for mankind"
    encoder = Encoder([text])
    coded = encoder.encodetext(text)
    decoded = encoder.decodeindexes(coded)

    print("Original text: %s" % text)
    print("Encoded text: " + str(coded))
    print("Decoded text: %s" % decoded)
    assert text == decoded


def test_consistent_encoding():
    """When two encoders are created with different corpus that share the same tokens, encoders are equal"""
    corpus1 = ["For the glory of mankind", "God's in his heaven. All's right with the world"]
    corpus2 = ["For the glory of God", "mankind's in his world. All's right with the heaven"]

    encoder1 = Encoder(corpus1)
    encoder2 = Encoder(corpus2)

    assert encoder1 == encoder2
