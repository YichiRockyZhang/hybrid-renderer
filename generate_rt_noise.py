from PIL import Image
import glob
import numpy as np
from imgaug import augmenters as iaa

def add_rt_noise(file_path):
    im = Image.open(file_path)
    im_arr = np.asarray(im)

    # gaussian noise
    # aug = iaa.AdditiveGaussianNoise(loc=0, scale=0.1*255)

    # poisson noise
    aug = iaa.AdditivePoissonNoise(lam=30.0, per_channel=False)
    im_arr = aug.augment_image(im_arr)

    # salt and pepper noise
    aug = iaa.SaltAndPepper(p=0.1)
    im_arr = aug.augment_image(im_arr)

    im = Image.fromarray(im_arr.astype(np.uint8)).convert('RGB')
    return im

print("[ADDING ARTIFICIAL NOISE TO IMAGES] Begin.")

decompress_dir = "/../mnt/d/hybrid-renderer-ds/"
jpg_files = glob.iglob(decompress_dir + '/**/*.color.jpg', recursive=True)
cutoff = len("color.jpg")

for file in jpg_files:
    print(file)
    noised = add_rt_noise(file)
    noised.save(file[:-cutoff] + "low_spp_rt.jpg")

print("[ADDING ARTIFICIAL NOISE TO IMAGES] Finished.")