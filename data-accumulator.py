from PIL import Image
import numpy as np
import imageio
filename = "../videoplayback.mp4"
video = imageio.get_reader(filename, "ffmpeg")

count = 0
for i, im in enumerate(video):
	if count == 20000:
		break
	if i % 10 == 0:
		count += 1
		print(count)
		img = Image.fromarray(np.uint8(im))
		off_x = 0.5 * (img.width - img.height)
		cropped = img.crop((off_x, 0, img.height + off_x, img.height))
		cropped.thumbnail((128, 128))
		cropped.convert('L').save("train/img" + str(count) + ".jpg")