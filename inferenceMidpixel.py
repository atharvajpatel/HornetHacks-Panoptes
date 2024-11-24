import torch
import cv2
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass, field

@dataclass
class PixelCoordinates:
    """Class to store middle pixel coordinates for each detection"""
    coordinates: Dict[str, List[Tuple[int, int]]] = field(default_factory=lambda: {
        'military_vehicle': [],
        'aircraft': [],
        'soldier': [],
        'civilian': [],
        'ordnance': []
    })

    def add_coordinate(self, class_name: str, coord: Tuple[int, int]):
        """Add a coordinate for a specific class"""
        if class_name in self.coordinates:
            self.coordinates[class_name].append(coord)

    def getDict(self) -> Dict[str, List[Tuple[int, int]]]:
        """Return the dictionary of all coordinates"""
        return self.coordinates

    def clear(self):
        """Clear all stored coordinates"""
        for key in self.coordinates:
            self.coordinates[key] = []

class MilitaryDetector:
    def __init__(self, model_path: str, input_file_path: str, conf_threshold: float = 0.25):
        """Initialize the military object detector.
        
        Args:
            model_path: Path to the YOLOv5 model file
            input_file_path: Path to the input image file
            conf_threshold: Confidence threshold for detections
        """
        # Store file path
        self.input_file_path = input_file_path
        
        # Initialize pixel coordinates tracker
        self.pixel_coordinates = PixelCoordinates()
        
        # Load model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model using torch hub
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        self.model.conf = conf_threshold
        self.model.to(self.device)
        
        # Class names - make sure these match your trained model's classes
        self.class_names = ['military_vehicle', 'aircraft', 'soldier', 'civilian', 'ordnance']
        
        # Colors (BGR)
        self.colors = {
            'military_vehicle': (0, 0, 255),
            'aircraft': (255, 0, 0),
            'soldier': (0, 255, 0),
            'civilian': (0, 255, 255),
            'ordnance': (255, 0, 255)
        }

    def calculate_middle_pixel(self, bbox: List[int]) -> Tuple[int, int]:
        """
        Calculate the middle pixel of a bounding box.
        
        Args:
            bbox: List of [x1, y1, x2, y2] coordinates
            
        Returns:
            Tuple of (middle_x, middle_y) coordinates
        """
        x1, y1, x2, y2 = bbox
        middle_x = int((x1 + x2) / 2)
        middle_y = int((y1 + y2) / 2)
        return (middle_x, middle_y)

    def detect(self, image_path: str = None, save_path: str = None) -> List:
        """Detect objects in an image and save the annotated result."""
        try:
            # Clear previous coordinates
            self.pixel_coordinates.clear()
            
            # Use provided image path or default to initialized path
            image_path = image_path or self.input_file_path
            
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image from {image_path}")
            
            # Run inference
            results = self.model(image)
            
            # Process results
            detections = []
            annotated_image = image.copy()
            
            # Get pandas DataFrame of results
            df = results.pandas().xyxy[0]
            
            if len(df) == 0:
                print("No detections found in the image.")
                if save_path:
                    cv2.imwrite(save_path, annotated_image)
                return []
            
            print("\n" + "="*50)
            print("MIDDLE PIXEL COORDINATES FOR DETECTED OBJECTS")
            print("="*50)
            
            for idx, det in df.iterrows():
                try:
                    x1, y1, x2, y2 = map(int, [det['xmin'], det['ymin'], det['xmax'], det['ymax']])
                    conf = float(det['confidence'])
                    class_idx = int(det['class'])
                    
                    # Verify class index is valid
                    if class_idx < 0 or class_idx >= len(self.class_names):
                        print(f"Warning: Invalid class index {class_idx}")
                        continue
                        
                    class_name = self.class_names[class_idx]
                    color = self.colors[class_name]
                    
                    # Calculate middle pixel
                    middle_x, middle_y = self.calculate_middle_pixel([x1, y1, x2, y2])
                    
                    # Store middle pixel coordinates
                    self.pixel_coordinates.add_coordinate(class_name, (middle_x, middle_y))
                    
                    # Print middle pixel coordinates prominently
                    print(f"\nDetection #{idx + 1}:")
                    print(f"Object Type: {class_name}")
                    print(f"Middle Pixel: (x={middle_x}, y={middle_y})")
                    print(f"Confidence: {conf:.2f}")
                    print("-" * 30)
                    
                    # Draw box
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw middle point
                    cv2.circle(annotated_image, (middle_x, middle_y), 4, color, -1)
                    
                    # Add label with middle pixel coordinates
                    label = f"{class_name} {conf:.2f} | Center: ({middle_x}, {middle_y})"
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(annotated_image, (x1, y1 - h - 5), (x1 + w + 5, y1), color, -1)
                    cv2.putText(annotated_image, label, (x1 + 3, y1 - 3), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                    # Store detection with middle pixel
                    detections.append({
                        'class': class_name,
                        'confidence': conf,
                        'bbox': [x1, y1, x2, y2],
                        'middle_pixel': (middle_x, middle_y)
                    })
                except Exception as e:
                    print(f"Warning: Error processing detection: {e}")
                    continue
            
            # Save the annotated image if save path provided
            if save_path:
                cv2.imwrite(save_path, annotated_image)
                print(f"\nSaved annotated image to: {save_path}")
            
            # Print summary
            print("\nSUMMARY:")
            print(f"Total detections: {len(detections)}")
            print("Middle pixels for all detections:")
            for idx, det in enumerate(detections, 1):
                print(f"{idx}. {det['class']}: ({det['middle_pixel'][0]}, {det['middle_pixel'][1]})")
            
            return detections
            
        except Exception as e:
            print(f"Error during detection: {str(e)}")
            return []

if __name__ == "__main__":
    try:
        # Initialize detector with input file path
        detector = MilitaryDetector(
            model_path='yolov5n.pt',
            input_file_path="aerial-soldiers2.jpg",
            conf_threshold=0.25
        )
        
        # Run detection
        detections = detector.detect(save_path="aerial-soldiers2-box.jpg")
        
        # Example of getting all stored coordinates
        all_coordinates = detector.pixel_coordinates.getDict()
        print("\nStored coordinates by class:")
        for class_name, coords in all_coordinates.items():
            print(f"{class_name}: {coords}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print(f"Full error details: ", e.__class__.__name__)