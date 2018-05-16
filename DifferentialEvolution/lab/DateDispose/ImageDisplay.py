import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import numpy as np
import os

InputDir = 'adv_samples/'
def get_image(indextemp=-1):
    global InputDir
    image_paths = sorted([os.path.join(InputDir, i) for i in os.listdir(InputDir)])

    if indextemp != -1:
        index = indextemp
    else:
        index = np.random.randint(len(image_paths))

    path = image_paths[index]
    return mpimg.imread(path)

# fig = plt.figure(figsize=(10, 1.8))
fig = plt.figure()
gs = gridspec.GridSpec(10, 10)
for i in range(100):
    if i//10 == i%10:
        image = get_image(i//10)
    else:
        image = mpimg.imread('%d.jpg'%i)
    ax = fig.add_subplot(gs[i//10, i%10])
    ax.imshow(image)
    ax.set_xticks([])
    ax.set_yticks([])
gs.tight_layout(fig)

plt.savefig('Display.png')