from io import BytesIO
from PIL import Image, ImageDraw,ImageFont
from model import OcrHandle
dbnet_max_size = 6000
short_size = 960 *1
ocrhandle = OcrHandle()
filename = "test/id.jpeg"
f = open(filename,"rb+")
img_bytes = b''.join(f.readlines())
img = Image.open(BytesIO(img_bytes))

try:
    if hasattr(img, '_getexif') and img._getexif() is not None:
        orientation = 274
        exif = dict(img._getexif().items())
        if orientation not in exif:
            exif[orientation] = 0
        if exif[orientation] == 3:
            img = img.rotate(180, expand=True)
        elif exif[orientation] == 6:
            img = img.rotate(270, expand=True)
        elif exif[orientation] == 8:
            img = img.rotate(90, expand=True)
except Exception as e:
    import pdb
    pdb.set_trace()
img = img.convert("RGB")
if short_size < 64:
    print("短边尺寸过小，请调整短边尺寸")

img_w, img_h = img.size
if max(img_w, img_h) * (short_size * 1.0 / min(img_w, img_h)) > dbnet_max_size:
    print("图片reize后长边过长，请调整短边尺寸")

res = ocrhandle.text_predict(img,rgb=True,short_size=short_size,angle_detect_num=30)
print(res)

