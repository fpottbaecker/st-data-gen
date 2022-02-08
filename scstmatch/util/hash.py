import hashlib


def sha256_for_file(filename):
    with open(filename, "rb") as file:
        digest = hashlib.sha256()
        for block in iter(lambda: file.read(4096), b""):
            digest.update(block)
        return digest.hexdigest()
