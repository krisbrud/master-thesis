#%%

# from PIL import Image, ImageSequence

# Open the GIF
# with Image.open("gifs/policy_image_jan16-4.gif") as im:
#     # Iterate over frames
#     for i, frame in enumerate(ImageSequence.Iterator(im)):
#         # Save each frame
#         # print(i, frame)
#         if i % 100 == 0:
#             # Show the frame to the user
#             # frame.show()
            
        # frame.save("frame%d.png" % i)
# %%
import matplotlib.pyplot as plt
from PIL import Image, ImageSequence

filename = "gifs/policy_image_jan16-4.gif" 

# Open the GIF
with Image.open(filename) as im:
    # Create a figure with 10 columns and 1 row
    fig, axes = plt.subplots(1, 10, figsize=(20, 5))
    # Iterate over the first ten frames
    for i, frame in enumerate(ImageSequence.Iterator(im)):
        if i == 10:
            break
        # Plot each frame in the corresponding subplot
        axes[i].imshow(frame)
        axes[i].set_aspect("equal")
        # axes[i].axis("off")

    plt.show()

    # Save the figure as "someframes.pdf"
    fig.savefig("someframes.pdf")
# %%

import matplotlib.pyplot as plt
from PIL import Image

# Open the GIF
with Image.open(filename) as im:
    # Create a figure with 10 columns and 1 row
    fig, axes = plt.subplots(1, 10, figsize=(20, 5))
    # Iterate over the first ten frames
    for i, frame in enumerate(ImageSequence.Iterator(im)):
        if i == 10:
            break
        # Plot each frame in the corresponding subplot
        axes[i].imshow(frame)
        axes[i].set_adjustable("box")
        axes[i].axis('off')
    # plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
# %%
