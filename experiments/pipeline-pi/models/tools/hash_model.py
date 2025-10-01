import hashlib, sys

p = sys.argv[1]  # pad naar model.tflite
h = hashlib.sha256()
with open(p, "rb") as f:
    for chunk in iter(lambda: f.read(8192), b""):
        h.update(chunk)
print(h.hexdigest())