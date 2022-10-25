'''
Constants and functions for data augmentation.

- APD
'''

import skimage
from skimage.transform import AffineTransform, EuclideanTransform, warp, rotate, rescale, resize
import albumentations as A
import numpy as np
import random
import cv2

augmentation_choices = [None,
                  'gaussian', 'localvar', 'poisson', 'salt', 'pepper', 's&p', 'speckle', 'affine', # skimage
                  'rain', 'shadow', 'flare', 'snow' # Albumentations
                  ]


def transform_image(transform, image, scale=None, rotation=None, shear=None, translation=(1.0, 1.0)):
    if transform == None or transform == 'None':
      return image
    elif transform == 'affine':
        transform = AffineTransform(
            scale=scale,
            rotation=rotation,
            shear=shear,
            translation=translation
#             translation=(random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)),
            )
        # return images with correct data types
        if type(image[0][0]) == np.uint16:
            return warp(image.astype(np.float32), transform.inverse).astype(np.uint16)
        else:
            return warp(image.astype(np.float32), transform.inverse).astype(np.uint8)
        
        
    elif transform == 'gaussian' or transform == 'speckle':
        # print(f"image dype before cvtColor: {image.dtype}")
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # print(f"image dype after cvtColor: {image.dtype}")
        image = (skimage.util.random_noise(image, mode='gaussian', var=0.05)*255).astype(np.uint8)
        return image # cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif transform in ['gaussian', 'localvar', 'poisson', 'salt', 'pepper', 's&p', 'speckle']:
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = (skimage.util.random_noise(image, mode=transform)*255).astype(np.uint8)
        return image
    elif transform == 'rain':
        t = A.Compose([A.RandomRain(brightness_coefficient=0.9, drop_width=1, blur_value=5, p=1)],)
        transformed = t(image=image)
        return transformed['image'] 
    elif transform == 'shadow':
        t = A.Compose([A.RandomShadow(num_shadows_lower=1, num_shadows_upper=1, shadow_dimension=5, shadow_roi=(0, 0.5, 1, 1), p=1)],)
        transformed = t(image=image)
        return transformed['image']
    elif transform == 'flare':
        t = A.Compose([A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0.5, p=1)],)
        transformed = t(image=image)
        return transformed['image']
    elif transform == 'snow':
        t = A.Compose([A.RandomSnow(brightness_coeff=2.5, snow_point_lower=0.3, snow_point_upper=0.5, p=1)],)
        transformed = t(image=image)
        return transformed['image']
    
    return None