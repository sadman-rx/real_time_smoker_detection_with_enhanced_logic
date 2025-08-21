import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import time
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import traceback
import gc
import collections
from picamera2 import Picamera2
import onnxruntime as ort
import tflite_runtime.interpreter as tflite
import math
from screeninfo import get_monitors
import matplotlib.pyplot as plt
from IPython.display import clear_output
import tkinter as tk
from tkinter import ttk
from datetime import datetime

# Global variables for display
root = None
panel = None

# Try to import ImageTk, fall back to matplotlib if not available
try:
    from PIL import ImageTk
    USE_TKINTER = True
except ImportError:
    print("ImageTk not available, falling back to matplotlib display")
    import matplotlib.pyplot as plt
    from IPython.display import clear_output
    USE_TKINTER = False

# Model path
YOLO_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'altf4_v2_smoking_detection_model.onnx')
MOVENET_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'movenet_multipose_lightning.tflite')

class SmokerLabel:
   CONFIDENT_SMOKER = "confident_smoker"
   POSSIBLE_SMOKER = "possible_smoker"
   NON_SMOKER = "nonSmoker"
   CIGARETTE = "cigarette"

# Create temp directory in the project folder
TEMP_DIR = os.path.join(os.path.dirname(__file__), 'temp')
if not os.path.exists(TEMP_DIR):
   os.makedirs(TEMP_DIR)

def get_temp_path():
   """Generate a temporary file path in our custom temp directory"""
   temp_name = f"frame_{time.time_ns()}.jpg"
   return os.path.join(TEMP_DIR, temp_name)

def cleanup_temp_files():
   """Clean up old temporary files"""
   if os.path.exists(TEMP_DIR):
       for file in os.listdir(TEMP_DIR):
           try:
               os.remove(os.path.join(TEMP_DIR, file))
           except Exception:
               pass

# Global cache for models
_model_cache = {
   'object_model': None,
   'landmark_model': None,
   'initialized': False
}
# Global reusable arrays and objects
_reusable_cache = {
   'person_colors': [
       (255, 0, 0),    # Red
       (0, 255, 0),    # Green
       (0, 0, 255),    # Blue
       (255, 255, 0),  # Yellow
       (255, 0, 255),  # Magenta
       (0, 255, 255),  # Cyan
   ],
   'persons_boxes': [],
   'cigarette_boxes': [],
   'final_persons': [],
   'final_cigarettes': [],
   'person_landmarks': [],
   'annotated_image': None
}
def load_models():
    """Load both ONNX and TFLite models"""
    global _model_cache
   
    if not _model_cache['initialized']:
        try:
            # Load YOLO ONNX model
            yolo_path = os.path.join(os.path.dirname(__file__), YOLO_MODEL_PATH)
            providers = ['CPUExecutionProvider']  # Use CPU provider for Raspberry Pi
            _model_cache['object_model'] = ort.InferenceSession(yolo_path, providers=providers)
           
            # Load MoveNet TFLite model
            movenet_path = os.path.join(os.path.dirname(__file__), MOVENET_MODEL_PATH)
            interpreter = tflite.Interpreter(model_path=movenet_path)
            interpreter.allocate_tensors()
           
            # Get input and output details for MoveNet
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            _model_cache['input_details'] = input_details
            _model_cache['output_details'] = output_details
            _model_cache['landmark_model'] = interpreter
           
            _model_cache['initialized'] = True
            print("Models loaded successfully")
            print(f"YOLO Model: {YOLO_MODEL_PATH}")
            print(f"MoveNet Model: {MOVENET_MODEL_PATH}")
           
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            traceback.print_exc()
            return None, None
   
    return _model_cache['object_model'], _model_cache['landmark_model']

def preprocess_image(image_path):
    """Preprocess image for movenet model input"""
    try:
        # Read image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to read image from {image_path}")
        else:
            image = image_path.copy()
       
        # Convert RGBA to RGB if needed
        if image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
           
        # Get original dimensions
        orig_height, orig_width = image.shape[:2]
        
        pad_factor = int((orig_width - orig_height))
        
        # Apply white padding (255 for each channel)
        padded_image = cv2.copyMakeBorder(
            image,
            top= 0,
            bottom=pad_factor,
            left=0,
            right=0,
            borderType=cv2.BORDER_CONSTANT,
            value=(255, 255, 255)
            )
        
        # Convert to PIL Image for consistent resizing
        pil_image = Image.fromarray(padded_image)
       
        # Resize using PIL (using old-style constants)
        input_size = 256  # Standard size for MoveNet
        resized_image = pil_image.resize((input_size, input_size), Image.NEAREST)
       
        # Convert back to numpy array and ensure uint8
        input_image = np.array(resized_image, dtype=np.uint8)
       
        # Add batch dimension and ensure uint8
        input_image = np.expand_dims(input_image, axis=0).astype(np.uint8)
       
        # Calculate scale factors
        movenet_image_scale = orig_width / input_size
       
        return input_image, orig_height, orig_width, movenet_image_scale
       
    except Exception as e:
        print(f"Error in preprocess_image: {e}")
        traceback.print_exc()
        return None, (1, 1), (0, 0), 0, 0, False


def calculate_iou(box1, box2):
   """Calculate Intersection over Union of two boxes"""
   x1_inter = max(box1[0], box2[0])
   y1_inter = max(box1[1], box2[1])
   x2_inter = min(box1[2], box2[2])
   y2_inter = min(box1[3], box2[3])

   inter_area = max(0, x2_inter - x1_inter + 1) * max(0, y2_inter - y1_inter + 1)
   box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
   box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

   iou = inter_area / (box1_area + box2_area - inter_area)
   return iou

def non_max_suppression(boxes, iou_threshold=0.5):
   """Perform Non-Maximum Suppression on bounding boxes"""
   if not boxes:
       return []

   boxes = sorted(boxes, key=lambda x: x["confidence"], reverse=True)
   filtered_boxes = []

   while boxes:
       best_box = boxes.pop(0)
       filtered_boxes.append(best_box)

       boxes = [
           box for box in boxes
           if calculate_iou(best_box["coords"], box["coords"]) < iou_threshold
       ]
   return filtered_boxes


def calculate_center(box):
   """Calculate center point of a bounding box"""
   x1, y1, x2, y2 = box
   return (x1 + x2) / 2, (y1 + y2) / 2


def is_point_in_box(point, box, tolerance=20):
   """Check if a point is inside a bounding box with tolerance"""
   x, y = point
   x1, y1, x2, y2 = box
 
   x1 -= tolerance
   y1 -= tolerance
   x2 += tolerance
   y2 += tolerance
 
   return x1 <= x <= x2 and y1 <= y <= y2

def process_pose_landmarks(pose_keypoints, orig_height, orig_width, scale):
    """Process pose keypoints with proper error handling"""
    try:
        if pose_keypoints is None or len(pose_keypoints.shape) < 2:
            print("Invalid pose keypoints shape")
            return []

        processed_poses = []
        input_size = 256  # MoveNet input size
        
        

        # Handle different MoveNet output shapes
        if len(pose_keypoints.shape) == 4:  # Shape [1, 1, 17, 3]
            pose_keypoints = pose_keypoints.reshape(-1, 51)  # Reshape to [N, 51]

        if len(pose_keypoints.shape) == 2:  # Shape [N, 51]
            num_poses = pose_keypoints.shape[0]
            keypoints_per_pose = 17
            
            for i in range(num_poses):
                pose = pose_keypoints[i]  # Get single pose
                keypoints = []
                
                for j in range(keypoints_per_pose):
                    # Get coordinates in 256x256 space
                    y = float(pose[j * 3]) * input_size  # y-coordinate
                    x = float(pose[j * 3 + 1]) * input_size  # x-coordinate
                    confidence = float(pose[j * 3 + 2])  # confidence score
                    
                    # Scale back to original dimensions
                    x = x * scale
                    y = y * scale
                    
                    # Remove padding effect (clamp y to original height)
                    y = min(y, orig_height - 1)
                    
                    # Ensure within image bounds
                    x = max(0, min(x, orig_width - 1))
                    y = max(0, min(y, orig_height - 1))
                    
                    keypoints.append({
                        "id": j,
                        "coords": (int(x), int(y)),
                        "confidence": confidence
                    })
                
                if keypoints:  # Only add if we have valid keypoints
                    processed_poses.append({
                        "landmarks": keypoints,
                        "assigned": False
                    })

        elif len(pose_keypoints.shape) == 3:  # Shape [1, 6, 56]
            num_poses = pose_keypoints.shape[1]
            keypoints_per_pose = 17
            
            for i in range(num_poses):
                pose = pose_keypoints[0, i]  # Get single pose
                if np.any(pose):  # Check if pose contains any valid points
                    keypoints = []
                    for j in range(keypoints_per_pose):
                        y = float(pose[j * 3]) * input_size
                        x = float(pose[j * 3 + 1]) * input_size
                        confidence = float(pose[j * 3 + 2])
                        
                        # Scale back to original dimensions
                        x = x * scale
                        y = y * scale
                        
                        # Remove padding effect
                        y = min(y, orig_height - 1)
                        
                        # Ensure within image bounds
                        x = max(0, min(x, orig_width - 1))
                        y = max(0, min(y, orig_height - 1))
                        
                        keypoints.append({
                            "id": j,
                            "coords": (int(x), int(y)),
                            "confidence": confidence
                        })
                    
                    if keypoints:
                        processed_poses.append({
                            "landmarks": keypoints,
                            "assigned": False
                        })

        return processed_poses
            
       
    except Exception as e:
        print(f"Error processing landmarks: {str(e)}")
        return []

def analyze_pose(pose_data):
    """Analyze pose data to detect smoking-related gestures"""
    try:
        keypoints = pose_data['keypoints']
       
        # Get relevant keypoint positions
        nose = next((kp for kp in keypoints if kp['id'] == 0), None)
        left_wrist = next((kp for kp in keypoints if kp['id'] == 9), None)
        right_wrist = next((kp for kp in keypoints if kp['id'] == 10), None)
        left_elbow = next((kp for kp in keypoints if kp['id'] == 7), None)
        right_elbow = next((kp for kp in keypoints if kp['id'] == 8), None)
       
        if nose and (left_wrist or right_wrist) and (left_elbow or right_elbow):
            # Check if either hand is near the face
            nose_pos = nose['coords']
            smoking_gesture = False
           
            # Define distance thresholds
            face_threshold = 50  # pixels for hand-to-face distance
            elbow_angle_threshold = 60  # degrees for elbow bend
           
            # Check left arm
            if left_wrist and left_elbow:
                left_dist = np.sqrt(np.sum(np.square(np.array(nose_pos) - np.array(left_wrist['coords']))))
                if left_dist < face_threshold:
                    # Check elbow angle
                    left_shoulder = next((kp for kp in keypoints if kp['id'] == 5), None)
                    if left_shoulder:
                        angle = calculate_angle(
                            left_shoulder['coords'],
                            left_elbow['coords'],
                            left_wrist['coords']
                        )
                        if angle < elbow_angle_threshold:
                            smoking_gesture = True
           
            # Check right arm
            if right_wrist and right_elbow:
                right_dist = np.sqrt(np.sum(np.square(np.array(nose_pos) - np.array(right_wrist['coords']))))
                if right_dist < face_threshold:
                    # Check elbow angle
                    right_shoulder = next((kp for kp in keypoints if kp['id'] == 6), None)
                    if right_shoulder:
                        angle = calculate_angle(
                            right_shoulder['coords'],
                            right_elbow['coords'],
                            right_wrist['coords']
                        )
                        if angle < elbow_angle_threshold:
                            smoking_gesture = True
           
            return smoking_gesture
           
    except Exception as e:
        print(f"Error analyzing pose: {e}")
        traceback.print_exc()
       
    return False


def calculate_angle(a, b, c):
    """Calculate angle between three points"""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
   
    ba = a - b
    bc = c - b
   
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
   
    return np.degrees(angle)



def find_best_person_for_landmarks(landmarks, persons_boxes):
   """Find best matching person for landmarks"""
   best_person = None
   max_valid_landmarks = 0
 
   for person in persons_boxes:
       valid_landmarks = 0
       for landmark in landmarks:
           if is_point_in_box(landmark["coords"], person["coords"]):
               valid_landmarks += 1
     
       if valid_landmarks > max_valid_landmarks:
           max_valid_landmarks = valid_landmarks
           best_person = person
 
   return best_person if max_valid_landmarks > 0 else None

def has_overlapping_persons(current_person, all_persons):
   """Check if a person has any overlapping persons"""
   current_box = current_person["coords"]
   for other_person in all_persons:
       if other_person["coords"] != current_person["coords"]:  # Skip self
           if calculate_iou(current_box, other_person["coords"]) > 0:  # Using existing calculate_iou function
               return True
   return False

def calculate_helper_weight(person, cigarette, yolo_weight=0.6, landmark_weight=0.4):
   """Calculate helper weight for person-cigarette pair"""
   # Base YOLO weight
   weight = 0
   if person["yolo_smoker"]:
       weight += yolo_weight * person["confidence"]
 
   # Landmark weight
   min_landmark_distance = float('inf')
   cigarette_center = calculate_center(cigarette["coords"])
 
   # Check only landmarks 0 (nose), 9 (left wrist), 10 (right wrist)
   for landmark in person["landmarks"]:
       if landmark["id"] in [0, 9, 10]:
           distance = np.sqrt((cigarette_center[0] - landmark["coords"][0])**2 +
                            (cigarette_center[1] - landmark["coords"][1])**2)
           min_landmark_distance = min(min_landmark_distance, distance)
 
   if min_landmark_distance != float('inf'):
       # Convert distance to a 0-1 scale (closer = higher weight)
       landmark_score = 1 / (1 + min_landmark_distance/100)  # /100 to normalize distance
       weight += landmark_weight * landmark_score
 
   return weight

def run_yolo_detection(model, image):
    """Run YOLO detection using ONNX runtime"""
    try:
        # Convert PIL Image to numpy if needed
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image

        target_size = 640
        original_height, original_width = image_np.shape[:2]
        padding_flag = int((original_width - original_height))
        print("padding_flag: ", padding_flag)
        # Apply white padding (255 for each channel)
        padded_image = cv2.copyMakeBorder(
            image_np,
            top= 0,
            bottom=padding_flag,
            left=0,
            right=0,
            borderType=cv2.BORDER_CONSTANT,
            value=(255, 255, 255)
            )
        padded_image, compression_factor = safe_resize_image(padded_image, target_size, original_width)
        print( "compression factor:  ", compression_factor)

        # Normalize and prepare input tensor
        input_image = padded_image.astype(np.float32) / 255.0
        input_tensor = np.transpose(input_image, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, 0)


        # Get input name and run inference
        input_name = model.get_inputs()[0].name
        outputs = model.run(None, {input_name: input_tensor})

        # Process predictions
        predictions = outputs[0][0].transpose()  # Shape: (8400, 7)
       
        # Get confidence scores
        confidence_scores = predictions[:, 4:]
        max_conf = np.max(confidence_scores, axis=1)
        class_ids = np.argmax(confidence_scores, axis=1)
       
        # Filter by confidence threshold
        conf_threshold = 0.25
        valid_detections = max_conf > conf_threshold
       
        boxes = predictions[valid_detections, :4]
        scores = max_conf[valid_detections]
        classes = class_ids[valid_detections]

        def non_max_suppression(boxes, scores, iou_threshold, class_id):
            """Perform Non-Maximum Suppression"""
            if len(boxes) == 0:
                return np.array([])
               
            # Convert to corners format
            x = boxes[:, 0]
            y = boxes[:, 1]
            w = boxes[:, 2]
            h = boxes[:, 3]
           
            # Convert to corner coordinates
            x1 = x - w/2
            y1 = y - h/2
            x2 = x + w/2
            y2 = y + h/2
           
            # Compute areas
            areas = (x2 - x1) * (y2 - y1)
           
            # For person class, pre-filter small boxes
            if class_id == 1:  # person class
                area_threshold = np.max(areas) * 0.5  # Keep boxes at least 50% of max area
                valid_indices = areas >= area_threshold
                if np.any(valid_indices):
                    boxes = boxes[valid_indices]
                    scores = scores[valid_indices]
                    areas = areas[valid_indices]
                    x1 = x1[valid_indices]
                    y1 = y1[valid_indices]
                    x2 = x2[valid_indices]
                    y2 = y2[valid_indices]
           
            # Sort by confidence score
            order = scores.argsort()[::-1]
           
            keep = []
            while order.size > 0:
                i = order[0]
                keep.append(i)
               
                if order.size == 1:
                    break
                   
                # Compute IoU
                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])
               
                w = np.maximum(0.0, xx2 - xx1)
                h = np.maximum(0.0, yy2 - yy1)
                inter = w * h
               
                ovr = inter / (areas[i] + areas[order[1:]] - inter)
                inds = np.where(ovr <= iou_threshold)[0]
                order = order[inds + 1]
               
            return np.array(keep)
        # Apply NMS per class
        final_detections = []
       
        for class_id in np.unique(classes):
            class_mask = classes == class_id
            if not np.any(class_mask):
                continue
               
            class_boxes = boxes[class_mask]
            class_scores = scores[class_mask]
           
            # Apply NMS
            keep_indices = non_max_suppression(class_boxes, class_scores, 0.45, class_id)
           
            for idx in keep_indices:
                x, y, w, h = class_boxes[idx]
                score = class_scores[idx]
               
                # Convert to corner format
                x1 = x - w/2
                y1 = y - h/2
                x2 = x + w/2
                y2 = y + h/2
               
                # Create detection object
                detection = type('Detection', (), {
                    'cls': np.array([class_id]),
                    'conf': np.array([score]),
                    'xyxy': np.array([[x1, y1, x2, y2]])
                })
                final_detections.append(detection)
       
        # Create results object
        results = type('Results', (), {'boxes': final_detections})()
        return [results]
       
    except Exception as e:
        print(f"Error in YOLO detection: {str(e)}")
        traceback.print_exc()
        return None

def run_movenet(input_tensor, interpreter):
    """Run MoveNet model inference"""
    try:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
       
        # Check if dynamic shape model
        is_dynamic_shape = input_details[0]['shape_signature'][2] == -1
        if is_dynamic_shape:
            interpreter.resize_tensor_input(
                input_details[0]['index'],
                input_tensor.shape,
                strict=True)
            interpreter.allocate_tensors()
       
        # Set input tensor (already uint8 from preprocessing)
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        return interpreter.get_tensor(output_details[0]['index'])
    except Exception as e:
        print(f"Error in MoveNet detection: {e}")
        return None

def convert_cv2_to_pil(cv2_frame):
   """
   Convert a cv2 frame to PIL Image format that YOLO expects.
   Ensures consistent color format and channel ordering.
   """
   # Convert BGR to RGB
   rgb_frame = cv2.cvtColor(cv2_frame, cv2.COLOR_BGR2RGB)
 
   # Convert to PIL Image while preserving the color information
   pil_image = Image.fromarray(rgb_frame)

   return pil_image
def detect_smoking(image_input, object_model, landmark_model, show_display=True):
    """Main function to detect smoking in an image"""
    temp_path = None  # Initialize temp_path at the start
    scale = 1.0
    processed_poses = []  # Initialize processed_poses
    try:
        # Process YOLO detections in parallel
        def process_detection(box):
            # Extract box information properly from the box parameter
            class_id = int(box.cls.item())  # Use .item() to properly convert numpy scalar
            confidence = float(box.conf.item())  # Use .item() for confidence too
           
            # Get coordinates in YOLO's 640x640 space
            x1, y1, x2, y2 = map(float, box.xyxy[0])
           
            # Scale coordinates back to original image size
            orig_height, orig_width = image_np.shape[:2]
            scale = orig_width / 640

            # Apply scaling and convert to integers
            x1 = x1 * scale
            y1 = y1 * scale
            x2 = x2* scale
            y2 = y2 * scale

            y1 = min(y1, orig_height - 1)
            y2 = min(y2, orig_height - 1)

            # Convert to integers
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
           
            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, orig_width - 1))
            y1 = max(0, min(y1, orig_height - 1))
            x2 = max(0, min(x2, orig_width - 1))
            y2 = max(0, min(y2, orig_height - 1))
           
            if class_id == 2:  # Smoker class
                return {
                    "type": "person",
                    "data": {
                        "coords": (x1, y1, x2, y2),
                        "class_id": class_id,
                        "confidence": confidence,
                        "yolo_smoker": True,
                        "pipeline_smoker": False,
                        "final_label": None,
                        "overlapping_cigarette": None,
                        "overlapping_person": None,
                        "helper_weight": None,
                        "landmarks": []
                    }
                }
            elif class_id == 1:  # Non-smoker class
                return {
                    "type": "person",
                    "data": {
                        "coords": (x1, y1, x2, y2),
                        "class_id": class_id,
                        "confidence": confidence,
                        "yolo_smoker": False,
                        "pipeline_smoker": False,
                        "final_label": None,
                        "overlapping_cigarette": None,
                        "overlapping_person": None,
                        "helper_weight": None,
                        "landmarks": []
                    }
                }
            elif class_id == 0:  # Cigarette class
                return {
                    "type": "cigarette",
                    "data": {
                        "coords": (x1, y1, x2, y2),
                        "confidence": confidence
                    }
                }
            return None


        # Use cached models if not provided
        if object_model is None or landmark_model is None:
            object_model, landmark_model = load_models()


        # Handle different input types
        if isinstance(image_input, str):
            image = Image.open(image_input)
            image_np = np.array(image, dtype=np.uint8)
            print(f"Loaded image type: {image_np.dtype}")
        elif isinstance(image_input, Image.Image):
            image = image_input
            image_np = np.array(image, dtype=np.uint8)
            print(f"Converted PIL image type: {image_np.dtype}")
        else:
            raise ValueError("image_input must be either a file path or a PIL Image object")

        # Calculate scale factor early
        scale = get_scale_factor(image_np)

        # Clear previous arrays for reuse
        _reusable_cache['persons_boxes'].clear()
        _reusable_cache['cigarette_boxes'].clear()
        _reusable_cache['final_persons'].clear()
        _reusable_cache['final_cigarettes'].clear()
        _reusable_cache['person_landmarks'].clear()
   
        # Reuse the annotated image array
        if _reusable_cache['annotated_image'] is None or _reusable_cache['annotated_image'].shape != image_np.shape:
            _reusable_cache['annotated_image'] = image_np.copy()
        else:
            _reusable_cache['annotated_image'][:] = image_np

        # Create thread pool for parallel processing
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Run YOLO detection
            yolo_future = executor.submit(run_yolo_detection, object_model, image)
           
            # For pose estimation
            if isinstance(image_input, str):
                input_image, orig_height, orig_width, movenet_image_scale = preprocess_image(image_input)
            else:
                temp_path = get_temp_path()
                image.save(temp_path)
                input_image, orig_height, orig_width, movenet_image_scale = preprocess_image(temp_path)
           
            # Run pose estimation
            pose_future = executor.submit(run_movenet, input_image, landmark_model)
           
            # Wait for both results
            results = yolo_future.result()

            pose_keypoints = pose_future.result()

            # Process YOLO detections
            if results and results[0] and results[0].boxes:
                for box in results[0].boxes:
                    
                    detection = process_detection(box)
                    if detection:
                        if detection["type"] == "person":
                            _reusable_cache['persons_boxes'].append(detection["data"])
                        elif detection["type"] == "cigarette":
                            _reusable_cache['cigarette_boxes'].append(detection["data"])


            print("\n_reusable_cache['persons_boxes']: ", _reusable_cache['persons_boxes'])
            print("\n_reusable_cache['cigarette_boxes']: ", _reusable_cache['cigarette_boxes'])


            # Apply NMS and process poses
            _reusable_cache['persons_boxes'] = non_max_suppression(_reusable_cache['persons_boxes'])
            _reusable_cache['cigarette_boxes'] = non_max_suppression(_reusable_cache['cigarette_boxes'])


            # Process pose keypoints with proper structure handling
            if pose_keypoints is not None:
                # Handle MoveNet output shape [1, 1, 17, 3]
                if len(pose_keypoints.shape) == 4:
                    # Reshape to [N, 51] where N is number of poses
                    poses = pose_keypoints.reshape(-1, 51)
                else:
                    # If already in correct format, use as is
                    poses = pose_keypoints
                processed_poses = process_pose_landmarks(poses,
                                                      orig_height, orig_width,
                                                      movenet_image_scale
                                                    )
               
            # Process each person
            annotated_image = _reusable_cache['annotated_image']

            for person in _reusable_cache['persons_boxes']:
                # Process landmarks and assign poses
                x1, y1, x2, y2 = person["coords"]
                person_center = calculate_center((x1, y1, x2, y2))
                person["landmarks"] = []

                # Find best matching pose
                best_pose = None
                min_distance = float('inf')

                if processed_poses:  # Only process if we have poses
                    for pose in processed_poses:
                        if not pose.get("assigned", False):  # Check if pose is already assigned
                            valid_landmarks = []
                            for landmark in pose["landmarks"]:
                                if is_point_in_box(landmark["coords"], (x1, y1, x2, y2)):
                                    valid_landmarks.append(landmark)
                           
                            if valid_landmarks:
                                avg_x = sum(lm["coords"][0] for lm in valid_landmarks) / len(valid_landmarks)
                                avg_y = sum(lm["coords"][1] for lm in valid_landmarks) / len(valid_landmarks)
                                distance = np.sqrt((person_center[0] - avg_x)**2 + (person_center[1] - avg_y)**2)
                               
                                if distance < min_distance:
                                    min_distance = distance
                                    best_pose = pose

                    if best_pose is not None:
                        best_person = find_best_person_for_landmarks(best_pose["landmarks"], _reusable_cache['persons_boxes'])
                       
                        if best_person is None or best_person["coords"] == person["coords"]:
                            # Select specific landmarks (nose, wrists)
                            selected_landmarks = [lm for lm in best_pose["landmarks"] if lm["id"] in [0, 9, 10]]
                            person["landmarks"].extend(selected_landmarks)
                            best_pose["assigned"] = True


                _reusable_cache['person_landmarks'].append(person)
               
                # Draw annotations with proper scaling
                person_color = _reusable_cache['person_colors'][(len(_reusable_cache['person_landmarks']) - 1) % len(_reusable_cache['person_colors'])]
                scale = get_scale_factor(annotated_image)


                # Draw landmarks
                for landmark in person["landmarks"]:
                    point_radius = max(1, int(5 * scale))
                    cv2.circle(annotated_image, landmark["coords"], point_radius, person_color, -1)
                   
                    # Add scaled label
                    label_text = f"P{len(_reusable_cache['person_landmarks'])-1}L{landmark['id']} ({landmark['confidence']:.2f})"
                    font_scale = 1 * scale
                    font_thickness = max(1, int(scale))
                    cv2.putText(
                        annotated_image,
                        label_text,
                        (landmark["coords"][0] + 5, landmark["coords"][1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        person_color,
                        font_thickness
                    )

                # Draw person's bounding box center
                box_center = calculate_center(person["coords"])
                center_radius = max(1, int(5 * scale))
                cv2.circle(annotated_image, (int(box_center[0]), int(box_center[1])), center_radius, person_color, -1)
               
                # Add center label with scaling
                cv2.putText(
                    annotated_image,
                    f"P{len(_reusable_cache['person_landmarks'])-1}_center",
                    (int(box_center[0]) + 5, int(box_center[1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1 * scale,
                    person_color,
                    max(1, int(scale))
                )
                # Draw lines from center to landmarks
                line_thickness = max(2, int(scale))
                for landmark in person["landmarks"]:
                    cv2.line(
                        annotated_image,
                        (int(box_center[0]), int(box_center[1])),
                        landmark["coords"],
                        person_color,
                        line_thickness,
                        cv2.LINE_AA
                    )
        # Process persons and cigarettes
        persons_to_remove = set()
        cigarettes_to_remove = set()

        for person_idx, person in enumerate(_reusable_cache['persons_boxes']):
            # Check if person has overlapping persons
            has_overlapping_person = has_overlapping_persons(person, _reusable_cache['persons_boxes'])
       
            # Find all overlapping cigarettes for this person
            overlapping_cigarettes = []
            for cig_idx, cigarette in enumerate(_reusable_cache['cigarette_boxes']):
                if calculate_iou(person["coords"], cigarette["coords"]) > 0:
                    overlapping_cigarettes.append((cig_idx, cigarette))
       
            # Case 1: Person has no cigarettes
            if len(overlapping_cigarettes) == 0:
                # Add person to final_persons as non-smoker
                final_person = {
                    "coords": person["coords"],
                    "class_id": person["class_id"],
                    "confidence": person["confidence"],
                    "yolo_smoker": person["yolo_smoker"],
                    "pipeline_smoker": False,
                    "final_label": None,
                    "overlapping_cigarette": None,
                    "overlapping_person": None,
                    "helper_weight": None,
                    "landmarks": person["landmarks"],
                    "person_id": len(_reusable_cache['final_persons'])  # Assign unique ID
                }
                _reusable_cache['final_persons'].append(final_person)
                persons_to_remove.add(person_idx)
                continue
       
            # Only process further if person has no overlapping persons
            if not has_overlapping_person:
                # Case 2: Person has multiple cigarettes
                if len(overlapping_cigarettes) > 1:
                    # Sort by confidence and keep the best one
                    overlapping_cigarettes.sort(key=lambda x: x[1]["confidence"], reverse=True)
                    best_cigarette_idx, best_cigarette = overlapping_cigarettes[0]
               
                    # Create final person entry
                    final_person = {
                        "coords": person["coords"],
                        "class_id": person["class_id"],
                        "confidence": person["confidence"],
                        "yolo_smoker": person["yolo_smoker"],
                        "pipeline_smoker": True,
                        "final_label": None,
                        "overlapping_cigarette": best_cigarette,
                        "overlapping_person": None,
                        "helper_weight": None,
                        "landmarks": person["landmarks"],
                        "person_id": len(_reusable_cache['final_persons'])  # Assign unique ID
                    }
                    _reusable_cache['final_persons'].append(final_person)

                    # Add cigarette to final_cigarettes
                    _reusable_cache['final_cigarettes'].append({
                        "coords": best_cigarette["coords"],
                        "confidence": best_cigarette["confidence"],
                        "person_id": final_person["person_id"],  # Link to person
                        "cigarette_id": len(_reusable_cache['final_cigarettes'])    # Unique cigarette ID
                    })

                    # Mark all other cigarettes for removal
                    for cig_idx, _ in overlapping_cigarettes[1:]:
                        cigarettes_to_remove.add(cig_idx)
               
                    persons_to_remove.add(person_idx)

                # Case 3: Person has exactly one cigarette
                elif len(overlapping_cigarettes) == 1:
                    cig_idx, cigarette = overlapping_cigarettes[0]
               
                    # Create final person entry
                    final_person = {
                        "coords": person["coords"],
                        "class_id": person["class_id"],
                        "confidence": person["confidence"],
                        "yolo_smoker": person["yolo_smoker"],
                        "pipeline_smoker": True,
                        "final_label": None,
                        "overlapping_cigarette": cigarette,
                        "overlapping_person": None,
                        "helper_weight": None,
                        "landmarks": [],
                        "person_id": len(_reusable_cache['final_persons'])  # Assign unique ID
                    }
                    _reusable_cache['final_persons'].append(final_person)
               
                    # Add cigarette to final_cigarettes
                    _reusable_cache['final_cigarettes'].append({
                        "coords": cigarette["coords"],
                        "confidence": cigarette["confidence"],
                        "person_id": final_person["person_id"],  # Link to person
                        "cigarette_id": len(_reusable_cache['final_cigarettes'])    # Unique cigarette ID
                    })
               
                    cigarettes_to_remove.add(cig_idx)
                    persons_to_remove.add(person_idx)

        # Clean up: Remove processed items
        _reusable_cache['persons_boxes'] = [person for idx, person in enumerate(_reusable_cache['persons_boxes']) if idx not in persons_to_remove]
        _reusable_cache['cigarette_boxes'] = [cig for idx, cig in enumerate(_reusable_cache['cigarette_boxes']) if idx not in cigarettes_to_remove]

        overlapping_groups = []  # Will store groups of overlapping persons
        current_group = []

        # First, group overlapping persons
        processed_persons = set()
        for i, person1 in enumerate(_reusable_cache['persons_boxes']):
            if i not in processed_persons:
                current_group = [person1]
                processed_persons.add(i)
           
                # Keep checking for new overlaps until no more found
                idx = 0
                while idx < len(current_group):
                    current_person = current_group[idx]
               
                    # Check against all unprocessed persons
                    for j, person2 in enumerate(_reusable_cache['persons_boxes']):
                        if j not in processed_persons:
                            iou = calculate_iou(current_person["coords"], person2["coords"])
                            if iou > 0:
                                current_group.append(person2)
                                processed_persons.add(j)
                    idx += 1
           
                if len(current_group) > 1:
                    overlapping_groups.append(current_group)

        # Process each overlapping group
        persons_to_remove = set()
        cigarettes_to_remove = set()

        # Process each overlapping group
        for group in overlapping_groups:
            # Initialize person_cigarette_map for the group FIRST
            person_cigarette_map = {}
            for person in group:
                person_cigarette_map[id(person)] = []
           
            # Find cigarettes overlapping with each person
            for cigarette_idx, cigarette in enumerate(_reusable_cache['cigarette_boxes']):
                for person in group:
                    if calculate_iou(person["coords"], cigarette["coords"]) > 0:
                        person_cigarette_map[id(person)].append({
                            "cigarette": cigarette,
                            "cigarette_idx": cigarette_idx
                        })

            # 1. First collect all cigarettes overlapping with this group
            group_cigarettes = []
            for person in group:
                for cig_data in person_cigarette_map[id(person)]:
                    if cig_data["cigarette_idx"] not in [c["cigarette_idx"] for c in group_cigarettes]:
                        group_cigarettes.append({
                            "cigarette": next(cig for cig in _reusable_cache['cigarette_boxes'] if _reusable_cache['cigarette_boxes'].index(cig) == cig_data["cigarette_idx"]),
                            "cigarette_idx": cig_data["cigarette_idx"]
                        })
           
            # 2. Calculate weights for all person-cigarette pairs
            all_pairs = []
            for person in group:
                for cig_data in group_cigarettes:
                    weight = calculate_helper_weight(person, cig_data["cigarette"])
                    all_pairs.append({
                        "person": person,
                        "cigarette": cig_data["cigarette"],
                        "cigarette_idx": cig_data["cigarette_idx"],
                        "weight": weight
                    })

            # 3. Sort all pairs by weight to get best matches
            all_pairs.sort(key=lambda x: x["weight"], reverse=True)

            # 4. Assign cigarettes to persons (ensuring one cigarette per person)
            assigned_persons = set()
            assigned_cigarettes = set()
           
            for pair in all_pairs:
                person_id = id(pair["person"])
                cig_idx = pair["cigarette_idx"]
               
                # Skip if either person or cigarette already assigned
                if person_id in assigned_persons or cig_idx in assigned_cigarettes:
                    continue
               
                # Check if cigarette already in final_cigarettes
                cigarette_exists = any(
                    existing_cig["coords"] == pair["cigarette"]["coords"]
                    for existing_cig in _reusable_cache['final_cigarettes']
                )
                if not cigarette_exists:
                    # Add person as smoker
                    person_exists = any(
                        p["coords"] == pair["person"]["coords"]
                        for p in _reusable_cache['final_persons']
                    )
                    if not person_exists:
                        final_person = {
                            "coords": pair["person"]["coords"],
                            "class_id": pair["person"]["class_id"],
                            "confidence": pair["person"]["confidence"],
                            "yolo_smoker": pair["person"]["yolo_smoker"],
                            "pipeline_smoker": True,
                            "final_label": None,
                            "overlapping_cigarette": pair["cigarette"]["coords"],
                            "overlapping_person": True,
                            "helper_weight": pair["weight"],
                            "landmarks": pair["person"]["landmarks"],
                            "person_id": len(_reusable_cache['final_persons'])
                        }
                        _reusable_cache['final_persons'].append(final_person)
                       
                        # Add cigarette
                        _reusable_cache['final_cigarettes'].append({
                            "coords": pair["cigarette"]["coords"],
                            "confidence": pair["cigarette"]["confidence"],
                            "person_id": final_person["person_id"],
                            "cigarette_id": len(_reusable_cache['final_cigarettes'])
                        })
                       
                        assigned_persons.add(person_id)
                        assigned_cigarettes.add(cig_idx)
                        cigarettes_to_remove.add(cig_idx)
           
            # 5. Process remaining unassigned persons as non-smokers
            for person in group:
                if id(person) not in assigned_persons:
                    person_exists = any(
                        p["coords"] == person["coords"] for p in _reusable_cache['final_persons']
                    )

                    if not person_exists:
                        final_person = {
                            "coords": person["coords"],
                            "class_id": person["class_id"],
                            "confidence": person["confidence"],
                            "yolo_smoker": person["yolo_smoker"],
                            "pipeline_smoker": False,
                            "final_label": None,
                            "overlapping_cigarette": None,
                            "overlapping_person": True,
                            "helper_weight": None,
                            "landmarks": person["landmarks"],
                            "person_id": len(_reusable_cache['final_persons'])
                        }
                        _reusable_cache['final_persons'].append(final_person)
                   
                    persons_to_remove.add(id(person))
                   
        # Final cleanup
        _reusable_cache['persons_boxes'] = [person for person in _reusable_cache['persons_boxes'] if id(person) not in persons_to_remove]
        _reusable_cache['cigarette_boxes'] = [cig for idx, cig in enumerate(_reusable_cache['cigarette_boxes']) if idx not in cigarettes_to_remove]
   
        # Determine final labels for all persons in final_persons
        for person in _reusable_cache['final_persons']:
            # Case 1: Person has a unique cigarette (processed first in our overlapping logic)
            if (person["pipeline_smoker"] and
                person["overlapping_cigarette"] and
                not person["overlapping_person"]):
                person["final_label"] = SmokerLabel.CONFIDENT_SMOKER       
            # Case 2: Person has a cigarette (either unique or shared) and YOLO agrees
            elif person["pipeline_smoker"] and person["yolo_smoker"]:
                person["final_label"] = SmokerLabel.CONFIDENT_SMOKER
            # Case 3: Person has no cigarette and YOLO agrees they're not smoking
            elif not person["pipeline_smoker"] and not person["yolo_smoker"]:
                person["final_label"] = SmokerLabel.NON_SMOKER
            # Case 4: Pipeline and YOLO disagree
            elif person["pipeline_smoker"] != person["yolo_smoker"]:
                person["final_label"] = SmokerLabel.POSSIBLE_SMOKER
            # Case 5: Default case (should rarely happen)
            else:
                person["final_label"] = SmokerLabel.NON_SMOKER

        # Scale drawing parameters
        box_thickness = max(1, int(2 * scale))
        font_scale = max(0.3, 1* scale)
        font_thickness = max(1, int(2 * scale))

        # Draw persons with labels
        for person in _reusable_cache['final_persons']:
            label = person.get("final_label", SmokerLabel.NON_SMOKER)
           
            if label == SmokerLabel.CONFIDENT_SMOKER:
                color = (255, 0, 0)  # Red
            elif label == SmokerLabel.POSSIBLE_SMOKER:
                color = (0, 0, 255)  # Blue
            else:  # NON_SMOKER
                color = (0, 255, 0)  # Green
               
            x1, y1, x2, y2 = person["coords"]
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, box_thickness)
            # Calculate text position with scaled offset
            text_offset = max(10, int(10 * scale))
            cv2.putText(annotated_image, label, (x1, y1 - text_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)

        # Draw cigarettes with scaled annotations
        for cigarette in _reusable_cache['final_cigarettes']:
            x1, y1, x2, y2 = cigarette["coords"]
            cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 204, 204), box_thickness)              
            # Calculate text position with scaled offset
            text_offset = max(10, int(10 * scale))
            cv2.putText(annotated_image, "cigarette", (x1, y1 - text_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 204, 204), font_thickness)

        # Prepare metrics
        metrics = {
            'num_persons': len(_reusable_cache['final_persons']),
            'num_cigarettes': len(_reusable_cache['final_cigarettes']),
            'smokers': [p for p in _reusable_cache['final_persons'] if p.get("final_label") in
                    [SmokerLabel.CONFIDENT_SMOKER, SmokerLabel.POSSIBLE_SMOKER]]
        }
        # After all processing is done
        if show_display:
            try:
                # Ensure we're working with RGB format for display
                if len(annotated_image.shape) == 3 and annotated_image.shape[2] == 3:
                    display_image = annotated_image.copy()
                else:
                    print("Warning: Unexpected image format for display")
                    display_image = annotated_image
                
                # Display frame
                display_frame(display_image)                
                # Check if window was closed
                if root is None:
                    return None, None
                    
            except Exception as display_error:
                print(f"Warning: Display error: {str(display_error)}")
                # Try direct save as fallback
                try:
                    output_path = os.path.join(os.getcwd(), "current_frame.jpg")
                    cv2.imwrite(output_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
                    print(f"Saved frame directly as fallback to {output_path}")
                except Exception as save_error:
                    print(f"Fallback save failed: {str(save_error)}")
              
        return annotated_image, metrics

    except Exception as e:
        print(f"Error processing image: {str(e)}")
        raise

    finally:
        # Clean up temp file if it exists
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as cleanup_error:
                print(f"Warning: Error cleaning up temp file: {str(cleanup_error)}")


def initialize_display():
    """Initialize the display window"""
    global root, panel
    
    if USE_TKINTER:
        # Create main window if it doesn't exist
        if root is None:
            root = tk.Tk()
            root.title("Smoking Detection")
            
            # Create a label for displaying the image
            panel = ttk.Label(root)
            panel.pack(padx=10, pady=10)
            
            # Set window size
            root.geometry("800x600")
            
            # Configure style
            style = ttk.Style()
            style.configure("TLabel", background="black")
            
        return root, panel
    return None, None

def display_frame(frame):
    """Display frame using available display method"""
    global root, panel
    
    try:
        if USE_TKINTER:
            if root is None or panel is None:
                root, panel = initialize_display()
            
            # Convert to PIL Image
            image = Image.fromarray(frame)
            
            # Resize while maintaining aspect ratio
            display_size = (800, 600)  # Slightly smaller than window size
            image.thumbnail(display_size, Image.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image=image)
            
            # Update the panel image
            panel.configure(image=photo)
            panel.image = photo  # Keep a reference
            
            # Update the GUI
            root.update_idletasks()
            root.update()
        else:
            # Fallback to matplotlib display
            clear_output(wait=True)
            plt.figure(figsize=(8, 6))
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                plt.imshow(frame)
            plt.axis('off')
            plt.show()
            plt.close()
            
    except Exception as e:
        print(f"Error displaying frame: {e}")
        if root is not None:
            root.destroy()
            root = None
            panel = None

def draw_metrics(frame, fps=None, process_time=None):
    """Draw metrics on frame consistently"""
    try:
        # Convert to RGB if needed
        if isinstance(frame, np.ndarray) and frame.shape[-1] == 3:
            frame_rgb = frame.copy()
        else:
            frame_rgb = np.array(frame)
            
        # Add metrics text
        if fps is not None:
            metrics_text = f"FPS: {fps}"
            if process_time is not None:
                metrics_text += f" | Process Time: {process_time:.3f}s"
                
            # Print metrics to console instead of drawing on frame
            print(metrics_text, end='\r')
            
        return frame_rgb
        
    except Exception as e:
        print(f"Error drawing metrics: {e}")
        return frame


def get_scale_factor(image):
    """Calculate scale factor based on image size"""
    if isinstance(image, np.ndarray):
        height, width = image.shape[:2]
    else:  # PIL Image
        width, height = image.size

    # Use 4000x2252 as reference size (your static image size)
    reference_width = 1920
    reference_height = 1080
    
    # Calculate scale factor based on diagonal
    current_diagonal = np.sqrt(width**2 + height**2)
    reference_diagonal = np.sqrt(reference_width**2 + reference_height**2)
    
    return current_diagonal / reference_diagonal


def reset_camera(picam2):
    """Reset camera if it encounters an error"""
    try:
        picam2.close()
        time.sleep(1)  # Wait for camera to properly close
        return initialize_camera()
    except Exception as e:
        print(f"Error resetting camera: {e}")
        return None


def is_frame_valid(frame):
    """Check if the captured frame is valid"""
    return frame is not None and frame.size > 0


def initialize_camera():
    """Initialize Pi Camera v3 with optimal settings"""
    try:
        print("Starting camera initialization...")
        picam2 = Picamera2()
        
        # Configure camera with specific settings for better performance
        preview_config = picam2.create_preview_configuration(
            main={"size": (1920, 1080), "format": "RGB888"},
            buffer_count=1,  # Reduce buffer count for lower latency
            queue=False  # Disable frame queue for real-time processing
        )       
        # Apply configuration
        picam2.configure(preview_config)
        
        # Start the camera with specific controls
        picam2.set_controls({"FrameDurationLimits": (33333, 33333)})  # Target 30fps
        picam2.start()
        
        # Give camera time to initialize
        time.sleep(2)
        
        print("Camera initialization complete")
        return picam2
        
    except Exception as e:
        print(f"Camera initialization failed: {e}")
        traceback.print_exc()
        return None


def print_onnx_model_info(model_path):
    """Print ONNX model input and output information"""
    session = ort.InferenceSession(model_path)
   
    print("\nModel Inputs:")
    for input in session.get_inputs():
        print(f"Name: {input.name}")
        print(f"Shape: {input.shape}")
        print(f"Type: {input.type}")
   
    print("\nModel Outputs:")
    for output in session.get_outputs():
        print(f"Name: {output.name}")
        print(f"Shape: {output.shape}")
        print(f"Type: {output.type}")


# Add this to your main function before the detection loop:
print_onnx_model_info(YOLO_MODEL_PATH)


def safe_resize_image(image, target_size, original_width):
    """Safely resize an image using PIL"""  
    try:
        # If image is a numpy array with more than 3 dimensions, remove the batch dimension
        if isinstance(image, np.ndarray) and len(image.shape) > 3:
            image = np.squeeze(image)  # Remove batch dimension
           
        # Convert numpy array to PIL Image
        if isinstance(image, np.ndarray):
            # Convert float32 to uint8 if necessary
            if image.dtype == np.float32:
                image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)
       
        # Resize using PIL (using old-style constants)
        resized = image.resize((target_size, target_size), Image.BILINEAR)
        compression_factor = original_width / target_size       

        # Convert back to numpy array
        return np.array(resized), compression_factor      
       
    except Exception as e:
        print(f"Error in safe_resize_image: {str(e)}")
        traceback.print_exc()
        return None

def process_webcam_frame(frame, object_model, landmark_model):
    """Process webcam frame exactly like static image path"""
    try:
        # Convert frame to RGB (uint8) for both YOLO and MoveNet
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Create PIL Image for YOLO
        frame_pil = Image.fromarray(frame_rgb)        
        # Process frame
        processed_frame, metrics = detect_smoking(frame_pil, object_model, landmark_model, True)
        
    except Exception as e:
        print(f"Error processing frame: {e}")
        traceback.print_exc()
        return None

def main():
    print("\nInitializing Smoking Detection System...")
    
    # Initialize camera with retries
    max_retries = 3
    picam2 = None
    
    # Clean up old temp files at startup
    cleanup_temp_files()
    
    for attempt in range(max_retries):
        print(f"\nCamera initialization attempt {attempt + 1}/{max_retries}")
        picam2 = initialize_camera()
        if picam2:
            break
        time.sleep(1)
    
    if not picam2:
        print("Failed to initialize camera after multiple attempts. Exiting...")
        return

    print("\nLoading detection models...")
    object_model, landmark_model = load_models()

    if not object_model or not landmark_model:
        print("Failed to load models. Exiting...")
        picam2.close()
        return

    # Initialize processing variables
    frame_count = 0
    fps_start_time = time.time()
    fps = 0
    process_times = collections.deque(maxlen=10)
    last_save_time = time.time()
    MIN_SAVE_INTERVAL = 1.0
    running = True
    
    print("\nStarting detection loop...")
    try:
        while running:
            try:
                loop_start = time.time()
                
                # Capture frame
                frame = picam2.capture_array()
                
                if frame is None or not is_frame_valid(frame):
                    print("Invalid frame captured, retrying...")
                    time.sleep(0.1)
                    continue
                                
                # Process frame
                results = process_webcam_frame(frame, object_model, landmark_model)
                
                if results is not None:
                    # Calculate FPS
                    frame_count += 1
                    current_time = time.time()
                    
                    if current_time - fps_start_time >= 1.0:
                        fps = frame_count
                        print(f"FPS: {fps} | Processing Time: {(current_time - loop_start):.3f}s")
                        if results.get('has_detection', False):
                            print("Person/Cigarette detected!")
                        frame_count = 0
                        fps_start_time = current_time
                    
                    # Save processing time
                    process_time = time.time() - loop_start
                    process_times.append(process_time)
                
            except KeyboardInterrupt:
                print("\nStopping on keyboard interrupt...")
                running = False
            except Exception as e:
                print(f"Frame processing error: {str(e)}")
                traceback.print_exc()
                time.sleep(0.1)
                continue
            
            # Periodic cleanup of old temp files
            if frame_count % 300 == 0:  # Every 300 frames
                cleanup_temp_files()
                gc.collect()
                
    except Exception as e:
        print(f"\nFatal error: {str(e)}")
        traceback.print_exc()
    finally:
        print("\nCleaning up...")
        picam2.close()
        cleanup_temp_files()  # Final cleanup
        print("Cleanup complete")



if __name__ == "__main__":
   main()

