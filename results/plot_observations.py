# %%

import matplotlib.pyplot as plt
from PIL import Image, ImageSequence

filename = "gifs/policy_image_jan16-4.gif"

plt.style.use("ggplot")
#%%
# Open the GIF
with Image.open(filename) as im:
    # Create a figure with 10 columns and 1 row
    fig, axes = plt.subplots(1, 10, figsize=(20, 5))
    # Iterate over the first ten frames
    print(f"{len(list(ImageSequence.Iterator(im)))} frames")
    for i, frame in enumerate(ImageSequence.Iterator(im)):
        if i == 10:
            break
        # Plot each frame in the corresponding subplot
        axes[i].imshow(frame)
        axes[i].set_adjustable("box")
        axes[i].axis('off')
        title_offset = -0.3
        axes[i].set_title(f"t = {i}", y=title_offset)
        # axes[i].set_title(f"t = {i}")
    # plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
    fig.savefig("someframes.pdf")
# %%
import matplotlib.pyplot as plt
import math
from PIL import Image
import copy

ncols = 5
n_every = 50

# Open the GIF
with Image.open(filename) as im:
    # Get the total number of frames
    nframes = len(list(ImageSequence.Iterator(im)))
    # Calculate the number of rows
    nrows = math.ceil(nframes / (ncols * n_every))
    print("nrows", nrows)
    # Create a figure with ncols columns and nrows rows
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 20))
    # Flatten the axes array to make it 1-dimensional
    # axes = axes.ravel()
    # Iterate over all frames
    frames = ImageSequence.all_frames(im)  


    for i in range(nframes):
        if i % n_every == 0:
            frame = frames[i]
            print(frame)
            i_row = i // (ncols * n_every)
            i_col = (i // n_every) % ncols
            print(f"i = {i}")
            # j = i // n_every 
            # print(f"j = {j}")
            # Plot each frame in the corresponding subplot
            ax = axes[i_row][i_col]
            ax.imshow(frame)
            ax.set_adjustable("box")
            ax.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
# %%
