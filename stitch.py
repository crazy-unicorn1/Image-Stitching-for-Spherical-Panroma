import base64
from io import BytesIO

from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import cv2
from PIL import Image

from stitching import Stitcher
from stitching.feature_detector import FeatureDetector
from stitching.feature_matcher import FeatureMatcher
from stitching.subsetter import Subsetter
from stitching.camera_estimator import CameraEstimator
from stitching.camera_adjuster import CameraAdjuster
from stitching.camera_wave_corrector import WaveCorrector
from stitching.warper import Warper
from stitching.cropper import Cropper
from stitching.exposure_error_compensator import ExposureErrorCompensator
from stitching.seam_finder import SeamFinder
from stitching.blender import Blender
from stitching.timelapser import Timelapser
from stitching.images import Images

def decode_base64_image(img_str):
    decoded_img = base64.b64decode(img_str)
    byte_stream = BytesIO(decoded_img)
    img = Image.open(byte_stream)
    return img

def convert_to_array(img):
    img_array = np.array(img)
    if img_array.shape[2] == 3:
        img_array = img_array[:, :, ::-1]
    
    return img_array

def read_images_from_json(json_data, image_url_key='image'):
    '''
    Reads images from json object and returns array containing all images as base64 strings.

    json_data: json object containing image information
    image_url_key (optional): dictionary key to access url of image from json file. Defaults to 'image'
    '''
    # Array containing all images
    img_array = []
    try:
        # Read in images
        img_list = json_data
        # save image strings to array
        for idx, img_info in enumerate(img_list):
            # Retrieve image url using image_url_key
            img_string = img_info[image_url_key]
            if img_string.startswith("data:image/"):
                img_string = img_string.split(',')[1]
                img_array.append(img_string)
        return img_array
    except:
        return []

def read_images_fastapi_from_json(json_data, image_url_key='image'):
    '''
    Reads images from json object and returns array containing all images as base64 strings.

    json_data: json object containing image information
    image_url_key (optional): dictionary key to access url of image from json file. Defaults to 'image'
    '''
    # Array containing all images
    img_array = []
    try:
        # Read in images
        img_list = json_data
        # save image strings to array
        for idx, img_info in enumerate(img_list):
            # Retrieve image url using image_url_key
            img_string = img_info.image
            if img_string.startswith("data:image/"):
                img_string = img_string.split(',')[1]
                img_array.append(img_string)
        return img_array
    except:
        return []

def stitch(img_array, feature_detector='kaze', confidence_threshold=1, overlap_pct=0.3):
    '''
    Performs image stitching on an array of input images and returns a 2:1 equirectangular panoramic image encoded using base64
    The stitching module from https://github.com/OpenStitching/stitching serves as the basis for this. Certain classes have been
    modified to allow for more feature detectors and custom masking.

    img_array: array of input images represented as base64 strings
    feature_detector (optional): options currently are ['kaze', 'akaze', 'brisk']
    confidence_threshold (optional): Threshold value for keypoint 
    overlap_pct (optional): Fraction of overlap between images - to be used to avoid computing keypoints in non-overlapping regions
    '''
    # Use OpenCV to read in each file within inpath as a NumPy array representing an image
    try:
        if len(img_array) == 0:
            print('Input images were not read properly')
            return None , 500, {'error': 'Failed to read images. Expected 2 or more images encoded as string.', 'error_code': 501}
        images = []
        for img in img_array:
            img = decode_base64_image(img)
            img_array = convert_to_array(img)
            images.append(img_array)

        stitcher = Stitcher_360(detector=feature_detector, confidence_threshold=confidence_threshold, crop=True, compensator="gain")
        # Adapted from the Stitcher.stitch() function.
        panorama, status_code, json_response = stitcher.stitch_360(images, feature_detector, overlap_pct=overlap_pct)

        if status_code >= 500:
            return None, status_code, json_response
        # Crop the result image to be 2:1. Use the add_black_pixels function similarly to add black pixels instead of crop.
        #TODO: requires fixing
        #panorama_cropped = crop_to_aspect(panorama)
        # Convert from numpy array to PIL image
        panorama = panorama[:, :, ::-1]
        output_image = Image.fromarray(panorama)
        # Create a BytesIO object to store the image data
        image_io = BytesIO()
        # Save the Pillow Image object to the BytesIO object (you can choose the format)
        output_image.save(image_io, format='JPEG')  # You can change the format to other image formats
        # Get the raw image bytes from the BytesIO object
        image_bytes = image_io.getvalue()
        # Encode the raw bytes as base64
        encoded_string = base64.b64encode(image_bytes).decode("utf-8")

        return encoded_string, status_code, json_response

    except Exception as e:
        # StitcherError. Usually this means that not enough keypoints were detected to create an image.
        # It is recommended to take another set of pictures that contain at least 30% overlap so that keypoints can be better detected.
        # raise ValueError('There were not enough keypoints detected to create an image. It is recommended to take another set of pictures that contain at least 30 percent overlap so that keypoints can be better detected.')
        print(f"Stitching failed: {e}")
        return None, 501, {'error': 'Failed to stitch images.'}

def crop_to_aspect(image):
    '''
    Crops the panoramic image such that the aspect ratio is 2:1.
    Returns the cropped image.
    '''
    length, width, _ = image.shape
    # If the image is too long then trim the top and bottom
    if length * 2 > width:
        to_crop = length * 2 - width
        # Crop half on top
        image = image[to_crop//2:, :, :]
        # Crop other half on bottom
        image = image[:-(to_crop+1)//2, :, :]
    # If the image is too wide then trim the left and right
    elif length * 2 < width:
        to_crop = width - length * 2
        # Crop half to the left
        image = image[:, to_crop//2:, :]
        # Crop other half to the right
        image = image[:, :-(to_crop+1)//2, :]
    return image

def add_black_pixels(img):
  """
  Adds black pixels to an image to achieve a 2:1 aspect ratio.

  Args:
    img: input PIL image.
    output_img: output PIL image.
  """

  # Get original width and height
  width, height = img.size
  
  # Calculate new dimensions to achieve 2:1 aspect ratio
  if width >= 2 * height:
      new_height = int(width / 2)
      new_width = width
  else:
      new_height = height
      new_width = int(height * 2)

  # Calculate padding required
  top_padding = (new_height - height) // 2
  # bottom_padding = new_height - height - top_padding

  left_padding = (new_width - width) // 2
  # right_padding = new_width - width - left_padding

  # Create a new image with black background
  new_img = Image.new("RGB", (new_width, new_height), (0, 0, 0))

  # Paste the original image onto the new image with padding
  new_img.paste(img, (left_padding, top_padding))

  return new_img

class FeatureDetector_360(FeatureDetector):

    DETECTOR_CHOICES = OrderedDict()
    DETECTOR_CHOICES["orb"] = cv2.ORB.create
    DETECTOR_CHOICES["sift"] = cv2.SIFT_create
    DETECTOR_CHOICES["brisk"] = cv2.BRISK_create
    DETECTOR_CHOICES["akaze"] = cv2.AKAZE_create
    DETECTOR_CHOICES["kaze"] = cv2.KAZE_create

    def __init__(self, detector='kaze', **kwargs):
        self.detector = FeatureDetector_360.DETECTOR_CHOICES[detector](**kwargs)

    def detect_with_masks(self, imgs, masks):
        features = []
        for idx, img in enumerate(imgs):
            assert len(img.shape) == 3 and len(masks.shape) == 2
            # if not len(imgs) == len(masks):
            #     raise StitchingError("image and mask lists must be of same length")
            if not np.array_equal(img.shape[:2], masks.shape):
                raise ValueError(
                    f"Resolution of mask {idx+1} {masks.shape} does not match"
                    f" the resolution of image {idx+1} {img.shape[:2]}."
                )
            features.append(self.detect_features(img, mask=masks))
        return features
    
class Stitcher_360(Stitcher):
    def __init__(self, **kwargs):
        self.initialize_stitcher_360(**kwargs)

    def initialize_stitcher_360(self, **kwargs):
        self.settings = self.DEFAULT_SETTINGS.copy()
        self.validate_kwargs(kwargs)
        self.settings.update(kwargs)

        args = SimpleNamespace(**self.settings)
        self.medium_megapix = args.medium_megapix
        self.low_megapix = args.low_megapix
        self.final_megapix = args.final_megapix
        if args.detector in ("orb", "sift"):
            self.detector = FeatureDetector_360(args.detector, nfeatures=args.nfeatures)
        else:
            self.detector = FeatureDetector_360(args.detector)
        match_conf = FeatureMatcher.get_match_conf(args.match_conf, args.detector)
        self.matcher = FeatureMatcher(
            args.matcher_type,
            args.range_width,
            try_use_gpu=args.try_use_gpu,
            match_conf=match_conf,
        )
        self.subsetter = Subsetter(
            args.confidence_threshold, args.matches_graph_dot_file
        )
        self.camera_estimator = CameraEstimator(args.estimator)
        self.camera_adjuster = CameraAdjuster(
            args.adjuster, args.refinement_mask, args.confidence_threshold
        )
        self.wave_corrector = WaveCorrector(args.wave_correct_kind)
        self.warper = Warper(args.warper_type)
        self.cropper = Cropper(args.crop)
        self.compensator = ExposureErrorCompensator(
            args.compensator, args.nr_feeds, args.block_size
        )
        self.seam_finder = SeamFinder(args.finder)
        self.blender = Blender(args.blender_type, args.blend_strength)
        self.timelapser = Timelapser(args.timelapse, args.timelapse_prefix)

    def stitch_360(self, images, detector='kaze', overlap_pct=0.3):
        self.images = Images.of(
            images, self.medium_megapix, self.low_megapix, self.final_megapix
        )

        imgs = list(self.images.resize(Images.Resolution.MEDIUM))
        
        status_code = 200
        json_response = {'success': 'Images stitched successfully.', 'algorithm': 'akaze', 'featureless_images': []}
        error_json_response = {'error': 'Failed to stitch images.', 'error_code': 501, 'algorithm': 'akaze', 'featureless_images': []}

        # Create a feature mask
        feature_mask = np.ones(imgs[0].shape[:2], dtype=np.int8)
        top = int(overlap_pct * imgs[0].shape[0])
        bottom = int((1 - overlap_pct) * imgs[0].shape[0])
        left = int(overlap_pct * imgs[0].shape[1])
        right = int((1 - overlap_pct) * imgs[0].shape[1])
        feature_mask[top:bottom, left:right] = 0

        if detector == 'brisk':
            features = self.find_features_360(imgs, [])
        else:
            features = self.find_features_360(imgs, feature_mask)

        enough_features = True

        for idx, feature in enumerate(features):
            if len(feature.keypoints) == 0:
                enough_features = False
        
        if not enough_features:
            json_response['algorithm'] = 'orb'
            error_json_response['algorithm'] = 'orb'
            #try orb feature detector
            print("[INFO]Using alternative feature detector: orb")
            finder = FeatureDetector(detector='orb', nfeatures=500)
            features = [finder.detect_features(img) for img in imgs]

        featureless_images = [] 
        for idx, feature in enumerate(features):
            featureless_images.append({str(idx): len(feature.keypoints)})
            if len(feature.keypoints) == 0:
                enough_features = False
        
        enough_features = True

        for idx, feature in enumerate(features):
            if len(feature.keypoints) == 0:
                enough_features = False

        # No features found with orb or akaze, return to frontend for picture retaking
        if not enough_features:
            error_json_response['featureless_images'] = featureless_images
            return None, 501, error_json_response

        matches = self.match_features(features)
        imgs, features, matches = self.subset(imgs, features, matches)
        cameras = self.estimate_camera_parameters(features, matches)
        cameras = self.refine_camera_parameters(features, matches, cameras)
        cameras = self.perform_wave_correction(cameras)
        self.estimate_scale(cameras)

        imgs = self.resize_low_resolution(imgs)
        imgs, masks, corners, sizes = self.warp_low_resolution(imgs, cameras)

        #If cropping fails , we try without cropping
        try:
            self.cropper.prepare(imgs, masks, corners, sizes)
        except Exception as e:
            print(f"[INFO] Failed to crop: {e}")
            self.cropper = Cropper(False)
            self.cropper.prepare(imgs, masks, corners, sizes)

        imgs, masks, corners, sizes = self.crop_low_resolution(
            imgs, masks, corners, sizes
        )

        self.estimate_exposure_errors(corners, imgs, masks)
        seam_masks = self.find_seam_masks(imgs, corners, masks)

        imgs = self.resize_final_resolution()
        imgs, masks, corners, sizes = self.warp_final_resolution(imgs, cameras)
        imgs, masks, corners, sizes = self.crop_final_resolution(
            imgs, masks, corners, sizes
        )
        self.set_masks(masks)
        imgs = self.compensate_exposure_errors(corners, imgs)
        seam_masks = self.resize_seam_masks(seam_masks)

        self.initialize_composition(corners, sizes)
        self.blend_images(imgs, seam_masks, corners)
        return self.create_final_panorama(), status_code, json_response
    
    def find_features_360(self, imgs, feature_masks=[]):
        if len(feature_masks) == 0:
            return self.detector.detect(imgs)
        else:
            # feature_masks = Images.of(
            #     feature_masks, self.medium_megapix, self.low_megapix, self.final_megapix
            # )
            # feature_masks = list(feature_masks.resize(Images.Resolution.MEDIUM))
            # feature_masks = [Images.to_binary(mask) for mask in feature_masks]
            return self.detector.detect_with_masks(imgs, feature_masks)