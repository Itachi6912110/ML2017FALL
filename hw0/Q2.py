import PIL
import sys
from PIL import Image

im = Image.open(sys.argv[1])
width  = im.size[0]
length = im.size[1]

for w in range(0,width):
	for l in range(0,length):
		rgb = im.getpixel((w,l))
		rgb = (int(rgb[0]/2),int(rgb[1]/2),int(rgb[2]/2))
		im.putpixel((w,l),rgb)

im.show()
im.save("Q2.jpg")
