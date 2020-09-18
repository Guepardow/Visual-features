import os
import pandas as pd
import json
import xml.etree.ElementTree as ET
from tqdm import tqdm
import platform

from appearances import *
from affinity import *


if platform.system() == 'Linux':
    PATH_DATA = '/mnt/sda2/Dataset/'
elif platform.system() == 'Windows':
    PATH_DATA = 'D:/Dataset'

if not os.path.exists(PATH_DATA):
    raise ValueError(f"PATH_DATA was not found. Please update the file src/dataset.py")

def load_dataset(dataset, scene):
    """
    Load a dataset's bounding boxes filtered on a specific scene

    :param dataset: name of the dataset on interest
    :param scene: name of the scene of interest
    :return:
    """

    if dataset == 'WildTrack':
        return load_WildTrack(scene)
    elif dataset == 'DETRAC':
        return load_DETRAC(scene)
    elif dataset == 'MOT17':
        return load_MOT17(scene)
    elif dataset == 'UAVDT':
        return load_UAVDT(scene)
    else:
        raise ValueError(f"Dataset {dataset} is not recognized !")


class BoundingBoxes:
    """
    A class which contains all the information about bounding boxes
    """

    def __init__(self, height: int, width: int, framerate: int, dataset: str, first_frame: int,
                 xmin=None, xmax=None, ymin=None, ymax=None,
                 frames=None, objectIDs=None):
        """
        :param height: height of the images of the scene
        :param width: width of the images of the scene
        :param framerate: framerate of the video
        :param dataset: name of the dataset
        :param first_frame: number of the first frame
        :param xmin: numpy array of x-positions of the top-left boxes
        :param xmax: numpy array of x-positions of the bottom-right boxes
        :param ymin: numpy array of y-positions of the top-left boxes
        :param ymax: numpy array of y-positions of the bottom-right boxes
        :param frames: numpy array of frames
        :param objectIDs: numpy array of IDs of objects
        """

        self.height = height
        self.width = width
        self.framerate = framerate
        self.dataset = dataset
        self.first_frame = first_frame

        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

        self.frames = frames
        self.objectIDs = objectIDs
        self.features = None

    def __str__(self):
        return f"Dataset {self.dataset} \n \
        Number of unique frames : {len(np.unique(self.frames))}\n \
        Number of bounding boxes : {len(self.xmin)}\n \
        Average size of boxes : ({np.mean(self.xmax - self.xmin):.0f}, {np.mean(self.ymax - self.ymin):.0f})"

    def _open_frame_dataset(self, scene, frame):
        """
        Returns an image
        """

        if self.dataset == 'WildTrack':
            return open_frame_WildTrack(scene, frame)
        elif self.dataset == 'DETRAC':
            return open_frame_DETRAC(scene, frame)
        elif self.dataset == 'MOT17':
            return open_frame_MOT17(scene, frame)
        elif self.dataset == 'UAVDT':
            return open_frame_UAVDT(scene, frame)
        else:
            raise ValueError(f"Dataset {self.dataset} is not recognized !")

    def _compute_area(self):

        self.area = (self.xmax - self.xmin + 1) * (self.ymax - self.ymin + 1)
        return self

    def compute_features(self, scene, method_features):
        """
        Computes all the features vectors of each bounding boxes
        NB : it is advised to filter on a scene and on objects of interest
        """

        self.features = np.array([None] * len(self.xmin))
        unique_frames = np.unique(self.frames)
        for image_frame in tqdm(unique_frames):

            # Open the image
            image = self._open_frame_dataset(scene, image_frame)

            # print(f"Shape of image : {image.shape}")

            for i, (xmin, xmax, ymin, ymax, frame) in enumerate(zip(self.xmin, self.xmax, self.ymin, self.ymax, self.frames)):
                
                # print(f"Info on patch : {xmin}-{xmax}, {ymin}-{ymax}, {frame}")

                if frame == image_frame:
                    if method_features == "random":
                        self.features[i] = np.random.normal(size=100)
                    elif method_features == "bw":
                        self.features[i] = np.array(get_colorHistogram_BW(image[ymin:ymax, xmin:xmax]))
                    elif method_features == "colors":
                        self.features[i] = np.array(get_colorHistogram_BGR(image[ymin:ymax, xmin:xmax]))
                    elif method_features == "hog":
                        self.features[i] = np.array(get_HOG(image[ymin:ymax, xmin:xmax], self.dataset))
                    elif method_features == "vgg19":
                        self.features[i] = np.array(get_VGG19(image[ymin:ymax, xmin:xmax]))
                    elif method_features == "resnet18":
                        self.features[i] = np.array(get_ResNet18(image[ymin:ymax, xmin:xmax]))
                    elif method_features == "densenet121":
                        self.features[i] = np.array(get_Densenet121(image[ymin:ymax, xmin:xmax]))
                    elif method_features == "efficientnetB0":
                        self.features[i] = np.array(get_EfficientnetB0(image[ymin:ymax, xmin:xmax])) 
                    elif method_features == "osnetAINMarket":
                        self.features[i] = np.array(get_OSNet_AIN_Market(image[ymin:ymax, xmin:xmax]))   
                    elif method_features == "vehReid":
                        self.features[i] = np.array(get_VehReid(image[ymin:ymax, xmin:xmax]))  
                    else:
                        raise NotImplementedError(f"Method {method_features} is not recognized in the function compute_features() from dataset.py!")

        return self

    def get_noisy(self, sigma):

        n_elements = len(self.xmin)  # number of boxes

        # Get noisy version by adding a Gaussian white noise.
        # The larger the box is, the larger is the noise
        width_box = self.xmax - self.xmin
        height_box = self.ymax - self.ymin

        xmin = self.xmin + np.random.normal(scale=sigma * width_box,  size=n_elements).astype(int)
        xmax = self.xmax + np.random.normal(scale=sigma * width_box,  size=n_elements).astype(int)
        ymin = self.ymin + np.random.normal(scale=sigma * height_box, size=n_elements).astype(int)
        ymax = self.ymax + np.random.normal(scale=sigma * height_box, size=n_elements).astype(int)

        # Assure that the border are still inside the image
        xmin = np.clip(xmin, 0, self.width-1)
        xmax = np.clip(xmax, 0, self.width-1)
        ymin = np.clip(ymin, 0, self.height-1)
        ymax = np.clip(ymax, 0, self.height-1)

        # Assure that the bounding boxes' areas are not equal to zero
        xmax[(xmax == xmin) & (xmax != (self.width-1))] += 1
        xmin[(xmax == xmin) & (xmax == (self.width-1))] -= 1

        ymax[(ymax == ymin) & (ymax != (self.height-1))] += 1
        ymin[(ymax == ymin) & (ymax == (self.height-1))] -= 1

        # Assure that the min is strictly lower that the max
        # BE CAUTIOUS : here, to invert, a temporary variable is needed. That is why we use xmin instead of self.xmin ...
        self.xmin = np.minimum(xmin, xmax)
        self.xmax = np.maximum(xmin, xmax)
        self.ymin = np.minimum(ymin, ymax)
        self.ymax = np.maximum(ymin, ymax)

        return self

    def get_matching_score(self, affinity, step_sampling):

        counter = 0
        total = 0

        self._compute_area()  # For further analysis

        logs = []

        unique_frames = np.unique(self.frames)

        for frame1 in unique_frames:

            if frame1 <= max(unique_frames) - self.framerate * step_sampling:
                # DO NOT TAKE frame1 too large otherwise frame2 does not make any sense

                frame2 = frame1 + step_sampling * self.framerate

                # All objects in frame1 and frame2
                is_frame1 = self.frames == frame1
                is_frame2 = self.frames == frame2

                dict_features1 = {objectID: (feature, area) for objectID, feature, area in zip(self.objectIDs[is_frame1], self.features[is_frame1], self.area[is_frame1])}
                dict_features2 = {objectID: feature for objectID, feature in zip(self.objectIDs[is_frame2], self.features[is_frame2])}

                # Remove objects which are not in both images
                # If an object is not present is one frame, no comparison is allowed because it will be biased
                common_objects = list(set(dict_features1.keys()).intersection(dict_features2.keys())) 
                dict_features1 = {objectID: dict_features1[objectID] for objectID in common_objects}
                dict_features2 = {objectID: dict_features2[objectID] for objectID in common_objects}

                # Find the closest feature vector
                for objectID1, (feature1, area) in dict_features1.items():

                    if affinity == 'manhattan':
                        distances = get_manhattan(feature1, dict_features2)
                        best_objectID2 = min(distances, key=distances.get)
                        best_value = np.min(list(distances.values()))

                    elif affinity == 'euclidean':
                        distances = get_euclidean(feature1, dict_features2)
                        best_objectID2 = min(distances, key=distances.get)
                        best_value = np.min(list(distances.values()))

                    elif affinity == 'cosine':
                        similarities = get_cosine(feature1, dict_features2)
                        best_objectID2 = max(similarities, key=similarities.get)
                        best_value = np.max(list(similarities.values()))

                    elif affinity == 'rank1':
                        similarities = get_rank1(feature1, dict_features2)
                        best_objectID2 = max(similarities, key=similarities.get)
                        best_value = np.max(list(similarities.values()))

                    elif affinity == 'bhattacharyya':
                        distances = get_bhattacharyya(feature1, dict_features2)
                        best_objectID2 = min(distances, key=distances.get)
                        best_value = np.min(list(distances.values()))   

                    elif affinity == 'wasserstein':
                        distances = get_wasserstein(feature1, dict_features2)
                        best_objectID2 = min(distances, key=distances.get)
                        best_value = np.min(list(distances.values()))   

                    elif affinity == 'dotProduct':
                        similarities = get_dotProduct(feature1, dict_features2)
                        best_objectID2 = max(similarities, key=similarities.get)
                        best_value = np.max(list(similarities.values()))

                    else:
                        raise NotImplementedError(f"Distance method {affinity} is not recognized in the function get_matching_score() from dataset.py!")

                    # Let's save some logs ans some statistics
                    logs.append({'frame1': frame1, 'objectID1': objectID1, 'objectID2': best_objectID2,
                                 'value': best_value, 'area': area})
                    counter += objectID1 == best_objectID2
                    total += 1

        df_logs = pd.DataFrame(logs)            

        return df_logs, counter, total


def load_WildTrack(scene):
    """
    Returns
    -------
        A BoundingBoxes object with all the bounding boxes associated to all objects in all frames
    """

    # -- Characteristics of images in WildTrack --
    boundingboxes = BoundingBoxes(height=1080, width=1920, framerate=5, dataset='WildTrack', first_frame=0)

    # -- Load the bounding boxes information --

    # This path contains all json files 
    path_json = os.path.join(PATH_DATA, 'WildTrack', 'annotations_positions')
    
    # Read all json files <=> one file = one frame from 7 cameras. 400 files expected.
    json_files = [f for f in os.listdir(path_json) if os.path.isfile(os.path.join(path_json, f))]

    # Loop over all json files <=> loop over all 400 frames
    for json_file in json_files:
            
        with open(os.path.join(path_json, json_file), 'r') as f:
            datastore = json.load(f)  # read the json file
            
            # Loop over each individual
            for individual in datastore:
            
                # Views for one person
                views = individual['views']

                # Loop over 7 cameras
                for view in views:

                    # Check if it is detected
                    is_detected = np.any(np.array([view['xmax'], view['xmin'], view['ymax'], view['ymin']]) + 1)
                    if is_detected:
                        if boundingboxes.xmin is None:  # first iteration
                            boundingboxes.xmin = [view['xmin']]
                            boundingboxes.xmax = [view['xmax']]
                            boundingboxes.ymin = [view['ymin']]
                            boundingboxes.ymax = [view['ymax']]
                            boundingboxes.objectIDs = [individual['personID']]
                            boundingboxes.frames = [int(json_file[:-5])]
                            boundingboxes.scenes = ['C' + str(view['viewNum'] + 1)]

                        else:
                            boundingboxes.xmin = np.concatenate((boundingboxes.xmin, [view['xmin']]))
                            boundingboxes.xmax = np.concatenate((boundingboxes.xmax, [view['xmax']]))
                            boundingboxes.ymin = np.concatenate((boundingboxes.ymin, [view['ymin']]))
                            boundingboxes.ymax = np.concatenate((boundingboxes.ymax, [view['ymax']]))
                            boundingboxes.objectIDs = np.concatenate((boundingboxes.objectIDs, [individual['personID']]))
                            boundingboxes.frames = np.concatenate((boundingboxes.frames, [int(json_file[:-5])]))
                            boundingboxes.scenes = np.concatenate((boundingboxes.scenes, ['C' + str(view['viewNum']+1)]))

    # Put the boxes into the image ! 
    # In the data provided by WildTrack, some bounding boxes may be outside the frame of the image
    boundingboxes.xmin = np.clip(boundingboxes.xmin, 0, boundingboxes.width)
    boundingboxes.xmax = np.clip(boundingboxes.xmax, 0, boundingboxes.width)
    boundingboxes.ymin = np.clip(boundingboxes.ymin, 0, boundingboxes.height)
    boundingboxes.ymax = np.clip(boundingboxes.ymax, 0, boundingboxes.height)
    
    boundingboxes.frames = boundingboxes.frames
    boundingboxes.scenes = boundingboxes.scenes
    boundingboxes.objectIDs = boundingboxes.objectIDs
    boundingboxes.classes = np.array(['pedestrians'] * boundingboxes.frames.shape[0])  # There are only pedestrians

    # -- Filter on the scene of interest --
    idx_scene = boundingboxes.scenes == scene

    boundingboxes.xmin = boundingboxes.xmin[idx_scene]
    boundingboxes.xmax = boundingboxes.xmax[idx_scene]
    boundingboxes.ymin = boundingboxes.ymin[idx_scene]
    boundingboxes.ymax = boundingboxes.ymax[idx_scene]
    boundingboxes.frames = boundingboxes.frames[idx_scene]
    boundingboxes.objectIDs = boundingboxes.objectIDs[idx_scene]

    return boundingboxes
                  

def open_frame_WildTrack(scene, frame):

    image = cv2.imread(os.path.join(PATH_DATA, 'WildTrack', 'Image_subsets', scene, str(frame).zfill(8) + '.png'))
    return image


def load_MOT17(scene):
    """
    Returns
    -------
        A BoundingBoxes object with all the bounding boxes associated to all objects in all frames
    """

    # -- Characteristics of images in MOT17 --

    if scene in ['02', '04', '09', '10', '11', '13']:
        height, width = 1080, 1920
    elif scene in ['05']:
        height, width = 480, 640
    else:
        raise ValueError(f"Scene {scene} is not recognized !")

    boundingboxes = BoundingBoxes(height, width, framerate=1, dataset='MOT17', first_frame=1)
    
    # -- Load the bounding boxes information --

    # Open the file with bounding boxes
    data = pd.read_csv(os.path.join(PATH_DATA, 'MOT17', 'train', 'MOT17-' + scene + '-DPM', 'gt', 'gt.txt'),
                       names=['frames', 'objectIDs', 'xmin', 'ymin', 'width', 'height', 'x', 'y', 'z'])

    xmin = np.maximum(0, data['xmin'].values)
    xmax = np.maximum(0, data['xmin'].values) + data['width'].values
    ymin = np.maximum(0, data['ymin'])
    ymax = np.maximum(0, data['ymin']) + data['height'].values

    frames = data['frames'].values
    objectIDs = data['objectIDs'].values

    # -- Filter on the scene and on the objects of interest --

    idx_tokeep = (data['x'] != 0) & (data['y'] == 1)
    # According to the devkit, when x=0, it is a 0-marked GT ; when y != 1, it is not a pedestrian
    # Code following https://bitbucket.org/amilan/motchallenge-devkit/src/default/evaluateTracking.m

    boundingboxes.xmin = xmin[idx_tokeep]
    boundingboxes.xmax = xmax[idx_tokeep]
    boundingboxes.ymin = ymin[idx_tokeep]
    boundingboxes.ymax = ymax[idx_tokeep]
    boundingboxes.frames = frames[idx_tokeep]
    boundingboxes.objectIDs = objectIDs[idx_tokeep]

    return boundingboxes


def open_frame_MOT17(scene, frame):

    image = cv2.imread(os.path.join(PATH_DATA, 'MOT17', 'train', 'MOT17-' + scene + '-DPM', 'img1', str(frame).zfill(6) + '.jpg'))
    return image


def load_DETRAC(scene):
    # -- Characteristics of images in DETRAC --

    boundingboxes = BoundingBoxes(height=540, width=960, framerate=1, dataset='DETRAC', first_frame=1)

    # -- Load the bounding boxes information --
    xmin = []
    xmax = []
    ymin = []
    ymax = []
    objectIDs = []
    frames = []

    tree = ET.parse(os.path.join(PATH_DATA, 'DETRAC', 'DETRAC-Train-Annotations-XML-v3', 'MVI_' + scene + '_v3.xml'))
    root = tree.getroot()

    for node in root:
        if node.tag == 'frame':  # It is a frame
            
            # Get the ID of the frame
            frame = node.attrib['num']
            
            # Get the coordinates in that frame, and classes, and objectsID
            for vehicle in node[0]:
                objectID = vehicle.attrib['id']
                
                for bb in vehicle:  # for each bounding box, get coordinates
                    if bb.tag == 'box':
                        xmin.append(int(float(bb.attrib['left'])))
                        xmax.append(int(float(bb.attrib['left']) + float(bb.attrib['width'])))
                        
                        ymin.append(int(float(bb.attrib['top'])))
                        ymax.append(int(float(bb.attrib['top']) + float(bb.attrib['height'])))
                        
                        frames.append(int(frame))
                        objectIDs.append(int(objectID))
           
    boundingboxes.xmin = np.array(xmin)
    boundingboxes.xmax = np.array(xmax)
    boundingboxes.ymin = np.array(ymin)
    boundingboxes.ymax = np.array(ymax)
    boundingboxes.frames = np.array(frames)
    boundingboxes.objectIDs = np.array(objectIDs)

    return boundingboxes


def open_frame_DETRAC(scene, frame):
    image = cv2.imread(os.path.join(PATH_DATA, 'DETRAC', 'Insight-MVT_Annotation_Train', 'MVI_' + scene, 'img' + str(frame).zfill(5) + '.jpg'))
    return image


def load_UAVDT(scene):

    # -- Characteristics of images in UAVDT --

    boundingboxes = BoundingBoxes(height=540, width=1024, framerate=1, dataset='UAVDT', first_frame=1)

    # -- Load the bounding boxes information --

    # Open the file with bounding boxes
    data = pd.read_csv(os.path.join(PATH_DATA, 'UAVDT', 'UAV-benchmark-MOTD_v1.0', 'GT', 'M' + str(scene).zfill(4) + '_gt_whole.txt'),
                       names=['frames', 'objectIDs', 'xmin', 'ymin', 'width', 'height', 'outOfView', 'occlusion', 'category'])

    xmin = np.maximum(0, data['xmin'].values)
    xmax = np.maximum(0, data['xmin'].values) + data['width'].values
    ymin = np.maximum(0, data['ymin'].values)
    ymax = np.maximum(0, data['ymin'].values) + data['height'].values

    frames = data['frames'].values
    objectIDs = data['objectIDs'].values

    boundingboxes.xmin = xmin
    boundingboxes.xmax = xmax
    boundingboxes.ymin = ymin
    boundingboxes.ymax = ymax
    boundingboxes.frames = np.array(frames)
    boundingboxes.objectIDs = np.array(objectIDs)

    return boundingboxes


def open_frame_UAVDT(scene, frame):
    image = cv2.imread(os.path.join(PATH_DATA, 'UAVDT', 'UAV-benchmark-M', 'M' + str(scene).zfill(4), 'img' + str(frame).zfill(6) + '.jpg'))
    
    return image
