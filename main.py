import base64
import logging
import time
from bundle_adj import traverse
from features import matching
from stitch import convert_to_array, decode_base64_image, read_images_from_json
from PIL import Image
from io import BytesIO
from stitching.images import Images
from stitcher import stitch
import numpy as np
import cv2

#This flag is used for adding 2 additional images located at the top/bottom 
APPEND_TOP_BOT_IMAGES_FLAG = False

#This flag is used for filling in black pixels 
BLACK_PIXEL_FILL_FLAG = False

def multiband_blend(patches, shape, n_levels=5):
    """
    Use multi-band blending [1] to merge patches.

    References
    ----------
    [1] Brown, Matthew, and David G. Lowe. "Automatic panoramic image stitching
    using invariant features." International journal of computer vision 74.1
    (2007): 59-73.
    """
    weights = np.zeros(shape + (len(patches),), dtype="float32")

    for idx, (warped, _, irange) in enumerate(patches):
        yrange, xrange = irange  # unpack to make numpy happy
        weights[yrange, xrange, idx] = warped[..., 3]
    # find maximum patch for each pixel
    valid = np.sum(weights, axis=-1) > 0
    weights = weights.argmax(axis=-1)
    weights[~valid] = -1

    # initialize sharp high-res masks for the patches
    for idx, (warped, _, irange) in enumerate(patches):
        warped[..., 3] = weights[irange] == idx

    # blur outside the valid region to reduce artifacts
    #  but then remove invalid pixels - compute only the first time
    allmask = np.zeros(shape, dtype=bool)

    mosaic = np.zeros(shape + (3,), dtype="float32")
    prevs = [None] * len(patches)
    for lvl in range(n_levels):
        logging.debug(f"Blending level #{lvl + 1}")
        sigma = np.sqrt(2*lvl + 1.0)*4
        layer = np.zeros(shape + (3,), dtype="float32")  # delta for this level
        wsum = np.zeros(shape, dtype="float32")
        is_last = lvl == (n_levels - 1)

        for idx, (warped, mask, irange) in enumerate(patches):
            tile = prevs[idx] if prevs[idx] is not None else warped.copy()
            if not is_last:
                blurwarp = cv2.GaussianBlur(warped, (0, 0), sigma)
                tile[..., :3] -= blurwarp[..., :3]
                tile[..., 3] = blurwarp[..., 3]   # avoid sharp masks
                prevs[idx] = blurwarp

            layer[irange] += tile[..., :3] * tile[..., [3]]
            wsum[irange] += tile[..., 3]
            if lvl == 0:
                allmask[irange] |= ~mask

        layer[~allmask, :] = 0
        wsum[wsum == 0] = 1
        mosaic += layer / wsum[..., None]

    mosaic = np.clip(mosaic, 0.0, 1.0)   # avoid saturation artifacts
    return (255 * mosaic).astype(np.uint8)

def idx_to_keypoints(matches, kpts):
    """Replace keypoint indices with their coordinates."""
    def _i_to_k(match, kpt1, kpt2):
        return np.concatenate([kpt1[match[:, 0]], kpt2[match[:, 1]]],
                              axis=1)

    # homogeneous coordinates
    kpts = [np.concatenate([kp, np.ones((kp.shape[0], 1))], axis=1)
            for kp in kpts]

    # add match confidence (number of inliers)
    matches = {i: {j: (_i_to_k(m, kpts[i], kpts[j]), h, len(m))
                   for j, (m, h) in col.items()} for i, col in matches.items()}

    return matches

def append_top_bottom(panorama, top_img, bot_img):

    def cubemap_top_to_equirectangular(bottom_img, H, W):
        bottom_img_np = np.array(bottom_img)
        face_size1, face_size2 = bottom_img.size[0], bottom_img.size[1]
        print(bottom_img_np.size)
        
        equirectangular_img_np = np.zeros((W, H, 3), dtype=np.uint8)

        x, y = np.meshgrid(np.arange(H), np.arange(W))

        theta = (x / W) * 2 * np.pi - np.pi
        phi = (y / H) * np.pi - (np.pi / 2)

        X = np.cos(phi) * np.cos(theta)
        Y = np.sin(phi)
        Z = np.cos(phi) * np.sin(theta)

        mask = Y < 0
        
        with np.errstate(divide='ignore', invalid='ignore'):
            u = np.where(mask, X / np.abs(Y), 0)
            v = np.where(mask, Z / np.abs(Y), 0)

        u_img = ((u + 1) / 2 * (face_size2 - 1)).astype(int)
        v_img = ((v + 1) / 2 * (face_size1 - 1)).astype(int)

        u_img = np.clip(u_img, 0, face_size2 - 1)
        v_img = np.clip(v_img, 0, face_size1 - 1)
        print(max(v_img[mask]))
        print(max(u_img[mask]))
        print(bottom_img_np.shape)
        equirectangular_img_np[mask] = bottom_img_np[v_img[mask], u_img[mask]]

        equirectangular_img = Image.fromarray(equirectangular_img_np)

        return equirectangular_img

    def cubemap_bot_to_equirectangular(bot_img, H, W):
        bot_img_np = np.array(bot_img)
        face_size1, face_size2 = bot_img_np.shape[0], bot_img_np.shape[1]
        print(bot_img_np.shape)

        equirectangular_img_np = np.zeros((W, H, 3), dtype=np.uint8)
        print(equirectangular_img_np.shape)
        
        x, y = np.meshgrid(np.arange(W), np.arange(H))

        print(x)
        print(y)

        theta = (x / W) * 2 * np.pi - np.pi
        phi = (y / H) * np.pi - (np.pi / 2)

        X = np.cos(phi) * np.cos(theta)
        Y = np.sin(phi)
        Z = np.cos(phi) * np.sin(theta)

        mask = Y > 0
        
        with np.errstate(divide='ignore', invalid='ignore'):
            u = np.where(mask, X / np.abs(Y), 0)
            v = np.where(mask, Z / np.abs(Y), 0)

        u_img = ((u + 1) / 2 * (face_size1 - 1)).astype(int)
        v_img = ((v + 1) / 2 * (face_size2 - 1)).astype(int)

        u_img = np.clip(u_img, 0, face_size1 - 1)
        v_img = np.clip(v_img, 0, face_size2 - 1)

        equirectangular_img_np[mask] = bot_img_np[u_img[mask], v_img[mask]]

        equirectangular_img = Image.fromarray(equirectangular_img_np)

        return equirectangular_img
    
    
    #functions used to crop back pixels from equirectangular images
    def keep_bottom_half(image):
        width, height = image.size
        top = height // 4.25
        bottom_half_image = image.crop((0, height-top, width, height))
        return bottom_half_image

    def keep_top_half(image):
        width, height = image.size
        bottom_half_image = image.crop((0, 0, width, height/2))
        return bottom_half_image


    # Keep same width , increase height by 20% for bottom image
    new_W, new_H = int(panorama.size[0]), int(panorama.size[1]*0.2) , 
    equirectangular_bot = cubemap_bot_to_equirectangular(bot_img, new_H, new_W)
    bot_half = keep_bottom_half(equirectangular_bot)

    # Keep same width , increase height by 10% for top image
    new_W, new_H = int(panorama.size[0]), int(panorama.size[1]*0.1) 
    equirectangular_image = cubemap_top_to_equirectangular(top_img, new_H, new_W)
    top_half = keep_top_half(equirectangular_image)

    # Combine images
    widths, heights = zip(*(i.size for i in [top_half, bot_half, panorama]))
    max_width = max(widths)
    total_height = sum(heights)

    # Create final image and paste the 3 images into the final result
    final_im = Image.new('RGB', (max_width, total_height))
    final_im.paste(top_half, (0,0))
    final_im.paste(panorama, (0,top_half.size[1]))
    final_im.paste(bot_half, (0,top_half.size[1]+panorama.size[1]))
    
    return final_im

def append_top_bottom(panorama, top_img, bot_img):

    def cubemap_top_to_equirectangular(bottom_img, H, W):
        bottom_img_np = np.array(bottom_img)
        bottom_height, bottom_width = bottom_img_np.shape[:2]

        equirectangular_img_np = np.zeros((H, W, 3), dtype=np.uint8)

        x, y = np.meshgrid(np.arange(W), np.arange(H))

        theta = (x / W) * 2 * np.pi - np.pi
        phi = (y / H) * np.pi - (np.pi / 2)

        X = np.cos(phi) * np.cos(theta)
        Y = np.sin(phi)
        Z = np.cos(phi) * np.sin(theta)

        mask = Y < 0

        with np.errstate(divide='ignore', invalid='ignore'):
            u = np.where(mask, X / np.abs(Y), 0)
            v = np.where(mask, Z / np.abs(Y), 0)

        u_img = ((u + 1) / 2 * (bottom_width - 1)).astype(int)
        v_img = ((v + 1) / 2 * (bottom_height - 1)).astype(int)

        u_img = np.clip(u_img, 0, bottom_width - 1)
        v_img = np.clip(v_img, 0, bottom_height - 1)

        equirectangular_img_np[mask] = bottom_img_np[v_img[mask], u_img[mask]]

        equirectangular_img = Image.fromarray(equirectangular_img_np)

        return equirectangular_img
    
    def cubemap_bot_to_equirectangular(top_img, H, W):
        top_img_np = np.array(top_img)
        top_height, top_width = top_img_np.shape[:2]

        equirectangular_img_np = np.zeros((H, W, 3), dtype=np.uint8)

        x, y = np.meshgrid(np.arange(W), np.arange(H))

        theta = (x / W) * 2 * np.pi - np.pi
        phi = (y / H) * np.pi - (np.pi / 2)

        X = np.cos(phi) * np.sin(theta)
        Y = np.cos(phi) * np.cos(theta)
        Z = np.sin(phi)

        mask = Z > 0

        with np.errstate(divide='ignore', invalid='ignore'):
            u = np.where(mask, X / np.abs(Z), 0)
            v = np.where(mask, Y / np.abs(Z), 0)

        u_img = ((u + 1) / 2 * (top_width - 1)).astype(int)
        v_img = ((v + 1) / 2 * (top_height - 1)).astype(int)

        u_img = np.clip(u_img, 0, top_width - 1)
        v_img = np.clip(v_img, 0, top_height - 1)

        equirectangular_img_np[mask] = top_img_np[v_img[mask], u_img[mask]]

        equirectangular_img = Image.fromarray(equirectangular_img_np)

        return equirectangular_img

    def keep_bottom_half(image):
        width, height = image.size
        top = height // 4.25
        bottom_half_image = image.crop((0, height-top, width, height))
        return bottom_half_image

    def keep_top_half(image):
        width, height = image.size
        bottom_half_image = image.crop((0, 0, width, height/2))
        return bottom_half_image

    # Keep same width , increase height by 20% for bottom image
    new_W, new_H = int(panorama.size[0]), int(panorama.size[1]*0.2) , 
    equirectangular_bot = cubemap_bot_to_equirectangular(bot_img, new_H, new_W)
    bot_half = keep_bottom_half(equirectangular_bot)

    # Keep same width , increase height by 10% for top image
    new_W, new_H = int(panorama.size[0]), int(panorama.size[1]*0.1) 
    equirectangular_image = cubemap_top_to_equirectangular(top_img, new_H, new_W)
    top_half = keep_top_half(equirectangular_image)

    # Combine images
    widths, heights = zip(*(i.size for i in [top_half, bot_half, panorama]))
    max_width = max(widths)
    total_height = sum(heights)

    # Create final image and paste the 3 images into the final result
    final_im = Image.new('RGB', (max_width, total_height))
    final_im.paste(top_half, (0,0))
    final_im.paste(panorama, (0,top_half.size[1]))
    final_im.paste(bot_half, (0,top_half.size[1]+panorama.size[1]))
    
    return final_im

def black_pixels_fill(input_image):
    img = np.array(input_image)
    
    black_mask = (img[:,:,0] == 0) & (img[:,:,1] == 0) & (img[:,:,2] == 0)
    padded_img = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)
    
    kernel = np.ones((3, 3), dtype=np.uint8)
    sum_r = cv2.filter2D(padded_img[:,:,0], -1, kernel)[1:-1, 1:-1]
    sum_g = cv2.filter2D(padded_img[:,:,1], -1, kernel)[1:-1, 1:-1]
    sum_b = cv2.filter2D(padded_img[:,:,2], -1, kernel)[1:-1, 1:-1]
    
    non_black_mask = ~(padded_img == 0).all(axis=-1).astype(np.uint8)
    count_non_black_neighbors = cv2.filter2D(non_black_mask, -1, kernel)[1:-1, 1:-1] - black_mask.astype(np.uint8)
    
    count_non_black_neighbors = np.maximum(count_non_black_neighbors, 1)
    
    avg_r = sum_r / count_non_black_neighbors
    avg_g = sum_g / count_non_black_neighbors
    avg_b = sum_b / count_non_black_neighbors

    img[black_mask] = np.stack((avg_r[black_mask], avg_g[black_mask], avg_b[black_mask]), axis=-1)
 
    return Image.fromarray(img.astype(np.uint8))

def stitch_images(request):
    # Set CORS headers for preflight requests
    if request.method == 'OPTIONS':
        # Allows GET requests from any origin with the Content-Type
        # header and caches preflight response for an 3600s
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)

    # Set CORS headers for the main request
    headers = {
        'Access-Control-Allow-Origin': '*'
    }

    json_response = {'success': 'Images stitched successfully.', 'algorithm': 'sift', 'featureless_images': []}

    try:
        # Access request data
        json_data = request.get_json()
        # Read images from json
        input_images = read_images_from_json(json_data)

        # Handle the case when images are not properly sent
        if len(input_images) != len(json_data):
            return ({'error_message': f'Failed to read images. Expected {str(len(json_data))} , got {str(len(input_images))} images encoded as string.', 'error_code': 501}, 500, headers)
            # return {'error_message': f'Failed to read images. Expected {str(len(json_data))} , got {str(len(input_images))} images encoded as string.', 'error_code': 501}
        
        # Process images and generate panoramic output
        # Note: Read stitch.py to better understand the parameters. Modify the feature_detector parameter as needed.
        # output_image, status_code, json_response = stitch(input_images, feature_detector='akaze', confidence_threshold=1, overlap_pct=0.3)
        
        if len(input_images) == 0:
            print('Input images were not read properly')
            return ({'error_message': 'Failed to read images. Expected 2 or more images encoded as string.', 'error_code': 501}, 500, headers)
        
        images_orig = []

        for img in input_images:
            img = decode_base64_image(img)
            img_array = convert_to_array(img)
            images_orig.append(img_array)

        images = Images.of(images_orig[:-2])
        medium_imgs = list(images.resize(Images.Resolution.MEDIUM))

        # Generate features for each image
        kpts, matches = matching(medium_imgs)
    
        # Check number of keypoints , requires atleast 10 keypoints for matching
        for idx, x in enumerate(kpts):
            if len(x) < 10:
                return ({'error_message': f'Image {idx+1} has less than 10 keypoints.Please retake the photos.', 'error_code': 502}, 500, headers)
        
        # Image registration
        start = time.time()
        regions = traverse(medium_imgs, idx_to_keypoints(matches, kpts))
        logging.info(f"Image registration, time: {time.time() - start}")

        # Stitching
        start = time.time()
        panorama = stitch(regions, blender=multiband_blend, equalize=False, crop=True)
        logging.info(f"Built mosaic, time: {time.time() - start}")

        #Convert from BGR to RGB
        panorama = panorama[..., ::-1]
        # Convert to base64
        output_image = Image.fromarray(panorama)
        
        # If flag is set we append top and bottom images
        if APPEND_TOP_BOT_IMAGES_FLAG:
            top,bot = Image.fromarray(images_orig[-1]), Image.fromarray(images_orig[-2])
            output_image = append_top_bottom(output_image, top, bot)
            
        if BLACK_PIXEL_FILL_FLAG:
            output_image = black_pixels_fill(output_image)
            
        # Create a BytesIO object to store the image data
        image_io = BytesIO()
        # Save the Pillow Image object to the BytesIO object (you can choose the format)
        output_image.save(image_io, format='JPEG')
        # Get the raw image bytes from the BytesIO object
        image_bytes = image_io.getvalue()
        encoded_string = base64.b64encode(image_bytes).decode("utf-8")
        
    except Exception as e:
        print(e)
        error_message = str(e)
        if len(error_message) == 0:
            error_message = "Failed to stitch images."

        return ({'error_message': "Failed to stitch images.", 'error_code': 500}, 500, headers)

    # Return response with base64 string representation of panoramic output if successful. Otherwise response contains None
    json_response['output_image'] = "data:image/jpeg;base64," + encoded_string
    
    return (json_response, 200, headers)
    