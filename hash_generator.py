import hashlib

file_path = "doc/S3 bucket in AWSaa.pdf"

sha256 = hashlib.sha256()

with open(file_path, "rb") as f:
    while chunk := f.read(4096):
        sha256.update(chunk)

print("\nNEW HASH:\n")
print(sha256.hexdigest())