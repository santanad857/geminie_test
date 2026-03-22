"""Explore what get_drawings() returns for Document 2.0 (page index 0)."""
import fitz
import json

pdf_path = "./352 AA copy 2.pdf"
doc = fitz.open(pdf_path)
page = doc[0]  # First page of the single-page PDF

print(f"Page size: {page.rect}")
print(f"Rotation: {page.rotation}")
print()

drawings = page.get_drawings()
print(f"Total drawings: {len(drawings)}")
print()



# Just print the first 5 raw dicts so we can see the structure
for i, d in enumerate(drawings[:5]):
    print(f"--- Drawing {i} ---")
    for key, value in d.items():
        print(f"  {key}: {value}")
    print()
