"""Tests for SHA-256.

The vectors are the ones from FIPS 180-4 plus a long input that crosses a block
boundary, which is where a wrong padding or a wrong length counter shows up. A
hash that is wrong in a subtle way still looks like a hash, so there is no
point testing anything except the exact digest.
"""

from harness import Suite

from molla.sys.sha256 import sha256_hex


def _bytes(text: String) -> List[UInt8]:
    var out = List[UInt8]()
    var p = text.unsafe_ptr()
    for i in range(text.byte_length()):
        out.append(p.unsafe_load(i))
    return out^


def run(mut suite: Suite):
    suite.group("sha256")

    suite.check(
        sha256_hex(_bytes(String("")))
        == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "empty input",
    )
    suite.check(
        sha256_hex(_bytes(String("abc")))
        == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        "abc",
    )
    suite.check(
        sha256_hex(
            _bytes(
                String(
                    "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
                )
            )
        )
        == "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1",
        "56 bytes, one byte short of a padded block",
    )

    # 1000 bytes of 'a'. Enough blocks that a length counter kept in 32 bits
    # would still be right, and a buffer that failed to carry would not.
    var long = List[UInt8]()
    for _ in range(1000):
        long.append(97)
    suite.check(
        sha256_hex(long)
        == "41edece42d63e8d9bf515a9ba6932e1c20cbc9f5a5d134645adb5db1b9737ea3",
        "1000 bytes",
    )
