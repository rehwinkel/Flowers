from PIL import Image
import numpy as np

tileSize = 32
img = Image.open("train/img1.jpg")
tiles = []
tilex = img.width // tileSize * 2 - 1
for y in range(tilex):
	for x in range(tilex):
		offx = x * tileSize // 2
		offy = y * tileSize // 2
		tile = img.crop((offx, offy, tileSize + offx, tileSize + offy))
		#tile.save("tile" + str(y * tilex + x) + ".jpg")
		pixels = np.array(list(tile.tobytes())) * (1 / 255)
		print(pixels)