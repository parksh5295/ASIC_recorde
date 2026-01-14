from pathlib import Path


src = Path("C:\\ASIC-#\\Clustering_Method\\clustering_nomal_identify.py")
dst = Path("C:\\ASIC-#\\Clustering_Method\\clustering_nomal_identify_utf8.py")

with open(src, "r", encoding="utf-16") as f:
    content = f.read()

with open(dst, "w", encoding="utf-8") as f:
    f.write(content)

print("Encoding converted from UTF-16 to UTF-8")
