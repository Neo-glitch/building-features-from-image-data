get_ipython().getoutput("pip install pytesseract")


get_ipython().getoutput("pip install Pillow")


import pytesseract
from PIL import Image, ImageFilter

import numpy as np
import matplotlib.pyplot as plt
from skimage import color
import cv2
import io


seafood = Image.open("./datasets/images/seafood.png")

print(seafood.format, seafood.size, seafood.mode)  # mode is color channel


seafood # can be shown directly since a png file


# OCR
text = pytesseract.image_to_string(seafood, lang = "eng")

print(text)


# blur image and text ocr on it
seafood_blur = seafood.filter(ImageFilter.GaussianBlur(4))
seafood_blur


text_blur = pytesseract.image_to_string(seafood_blur, lang = "eng")
print(text_blur)


# flip image and see how tesseract works
seafood_flipped = seafood.transpose(Image.FLIP_TOP_BOTTOM)
seafood_flipped


text_flipped = pytesseract.image_to_string(seafood_flipped, lang="eng")
print(text_flipped)


# gets text and store as a pdf file
pdf_bytes = pytesseract.image_to_pdf_or_hocr("./datasets/images/seafood.png", extension = "pdf")


new_file = open("seafood.pdf", "wb")
new_file_byte_array = bytearray(pdf_bytes)  # bytes array of pdf file


new_file.write(new_file_byte_array)
new_file.close()































































































































































