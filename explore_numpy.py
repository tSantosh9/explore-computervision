import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

pic = Image.open('IMG_5468.jpeg')
print(type(pic))

pic_arr = np.asarray(pic)
print(pic_arr.shape)

# 0: Red, 1: Green, 2: Blue
# plt.imshow(pic_arr[:,:,0], cmap='gray')

# Show the image without the green color
# Set the value of green channel to 0
pic_red = pic_arr.copy()
pic_red[:,:,1] = 0
plt.imshow(pic_red)

plt.show()
