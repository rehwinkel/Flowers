from PIL import Image
import numpy as np

def tilizeImage(filename):
	tileSize = 32
	img = Image.open(filename)
	tiles = []
	tilex = img.width // tileSize * 2 - 1
	for y in range(tilex):
		for x in range(tilex):
			offx = x * tileSize // 2
			offy = y * tileSize // 2
			tile = img.crop((offx, offy, tileSize + offx, tileSize + offy))
			pixels = np.array(list(tile.tobytes())) * (1 / 255)
			tiles.append(pixels)
	tiles = np.array(tiles)
	return tiles