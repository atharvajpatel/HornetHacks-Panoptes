# some random trigonometry right now with rough identification
# labels the image and uh returns x, y, z of observer

import cv2
import numpy as np
from sklearn.neighbors import KDTree
import pandas as pd
from datetime import datetime
import math

class ZenithCelestialNavigator:
    def __init__(self):
        """
        Initialize the celestial navigation system with specific focus on Orion.
        """
        self.star_catalog = self._load_orion_catalog()
        self.orion_pattern = self._define_orion_pattern()
        self.star_tree = self._build_star_tree()
        
    def _load_orion_catalog(self):
        """
        Load catalog with Orion's major stars.
        Coordinates are in right ascension (RA) and declination (Dec).
        """
        orion_stars = {
            'Betelgeuse': (88.79, 7.41, 0.42),    # Alpha Orionis
            'Rigel': (78.63, -8.20, 0.12),        # Beta Orionis
            'Bellatrix': (81.28, 6.35, 1.64),     # Gamma Orionis
            'Mintaka': (83.00, -0.30, 2.23),      # Delta Orionis
            'Alnilam': (84.05, -1.20, 1.69),      # Epsilon Orionis
            'Alnitak': (85.19, -1.94, 1.88),      # Zeta Orionis
            'Saiph': (86.94, -9.67, 2.06)         # Kappa Orionis
        }
        
        df = pd.DataFrame([
            {'name': name, 'ra': coords[0], 'dec': coords[1], 'magnitude': coords[2]}
            for name, coords in orion_stars.items()
        ])
        return df

    def _define_orion_pattern(self):
        """
        Define the relative pattern of Orion's stars.
        Returns normalized vectors between key stars.
        """
        # Define key star patterns in Orion (simplified relative positions)
        return {
            'belt': [
                {'start': 'Mintaka', 'end': 'Alnilam'},
                {'start': 'Alnilam', 'end': 'Alnitak'}
            ],
            'shoulders': [
                {'start': 'Betelgeuse', 'end': 'Bellatrix'}
            ],
            'feet': [
                {'start': 'Rigel', 'end': 'Saiph'}
            ],
            'body': [
                {'start': 'Alnilam', 'end': 'Betelgeuse'},
                {'start': 'Alnilam', 'end': 'Rigel'}
            ]
        }

    def _build_star_tree(self):
        """Build KD-tree for star pattern matching."""
        coords = np.column_stack((self.star_catalog['ra'], self.star_catalog['dec']))
        return KDTree(coords)
    
    
    def detect_stars_improved(self, image_path, min_brightness=180, min_area=3, max_area=150):
        """
        Improved star detection specifically tuned for night sky images.
        
        Args:
            image_path: Path to the image file
            min_brightness: Minimum pixel brightness to consider (0-255)
            min_area: Minimum area of star blob
            max_area: Maximum area of star blob
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise Exception("Failed to load image")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply local contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        
        # Binary threshold
        _, binary = cv2.threshold(enhanced, min_brightness, 255, cv2.THRESH_BINARY)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        stars = []
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if min_area <= area <= max_area:
                x = centroids[i][0]
                y = centroids[i][1]
                # Get maximum brightness in the component
                mask = (labels == i).astype(np.uint8)
                brightness = np.max(enhanced * mask)
                stars.append((x, y, brightness))
        
        # Sort by brightness
        return sorted(stars, key=lambda x: x[2], reverse=True)

    def detect_stars(self, image_path, threshold=128):
        """
        Enhanced star detection with focus on Orion-like patterns.
        """
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise Exception("Failed to load image")
            
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img = clahe.apply(img)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Find and filter star contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        stars = []
        for contour in contours:
            # Calculate centroid
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                # Get brightness and area
                brightness = img[cy, cx]
                area = cv2.contourArea(contour)
                # Filter by brightness and size
                if brightness > threshold and 5 < area < 100:  # Adjust these thresholds
                    stars.append((cx, cy, brightness))
        
        return sorted(stars, key=lambda x: x[2], reverse=True)
    


    def identify_orion_improved(self, stars, image_shape, belt_width_range=(0.15, 0.4)):
        """
        Improved Orion identification with parameters tuned for typical night sky photos.
        
        Args:
            stars: List of (x, y, brightness) tuples
            image_shape: Shape of the original image
            belt_width_range: (min, max) range for belt width as fraction of image width
        """
        if len(stars) < 5:
            return None
            
        height, width = image_shape[:2]
        bright_stars = stars[:20]  # Consider more stars for better detection
        
        def calc_distance(p1, p2):
            return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
        
        def is_roughly_collinear(p1, p2, p3, tolerance_degrees=20):  # Increased tolerance
            v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
            v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
            
            dot_product = np.dot(v1, v2)
            norms = np.linalg.norm(v1) * np.linalg.norm(v2)
            
            if norms == 0:
                return False
                
            angle = np.degrees(np.arccos(np.clip(dot_product / norms, -1.0, 1.0)))
            return abs(180 - angle) < tolerance_degrees
        
        # Find belt candidates with relaxed constraints
        best_belt = None
        best_belt_score = float('inf')
        
        min_belt_width = belt_width_range[0] * width
        max_belt_width = belt_width_range[1] * width
        
        for i in range(len(bright_stars) - 2):
            for j in range(i + 1, len(bright_stars) - 1):
                for k in range(j + 1, len(bright_stars)):
                    p1 = np.array(bright_stars[i][:2])
                    p2 = np.array(bright_stars[j][:2])
                    p3 = np.array(bright_stars[k][:2])
                    
                    if not is_roughly_collinear(p1, p2, p3):
                        continue
                    
                    d1 = calc_distance(p1, p2)
                    d2 = calc_distance(p2, p3)
                    belt_length = d1 + d2
                    
                    # Check if belt length is within expected range
                    if not (min_belt_width <= belt_length <= max_belt_width):
                        continue
                    
                    # Check spacing ratio with more tolerance
                    spacing_ratio = d1/d2 if d2 > 0 else float('inf')
                    if not (0.4 < spacing_ratio < 2.5):  # More tolerant ratio
                        continue
                    
                    # Score based on collinearity and spacing
                    angle_score = abs(180 - np.degrees(np.arccos(np.clip(np.dot(
                        p3 - p2, p2 - p1) / (np.linalg.norm(p3 - p2) * np.linalg.norm(p2 - p1)),
                        -1.0, 1.0))))
                    spacing_score = abs(1 - spacing_ratio)
                    brightness_score = abs(bright_stars[i][2] - bright_stars[j][2]) + abs(bright_stars[j][2] - bright_stars[k][2])
                    total_score = angle_score + spacing_score + brightness_score * 0.1
                    
                    if total_score < best_belt_score:
                        best_belt_score = total_score
                        best_belt = {
                            'stars': [p1, p2, p3],
                            'score': total_score,
                            'brightness': [bright_stars[i][2], bright_stars[j][2], bright_stars[k][2]]
                        }
        
        if not best_belt or best_belt_score > 30:  # More tolerant threshold
            return None
        
        # Find Betelgeuse and Rigel with improved criteria
        belt_center = np.mean(best_belt['stars'], axis=0)
        belt_vector = best_belt['stars'][2] - best_belt['stars'][0]
        belt_normal = np.array([-belt_vector[1], belt_vector[0]])
        belt_length = np.linalg.norm(belt_vector)
        
        # Look for bright stars at expected distances
        candidates = []
        for star in bright_stars:
            if all(not np.allclose(star[:2], belt_star) for belt_star in best_belt['stars']):
                pos = np.array(star[:2])
                dist_to_belt = abs(np.dot(pos - belt_center, belt_normal)) / np.linalg.norm(belt_normal)
                relative_dist = dist_to_belt / belt_length
                # Expect Betelgeuse and Rigel to be 0.8-1.5 times the belt length away
                if 0.6 < relative_dist < 1.8:
                    candidates.append((star, dist_to_belt, np.dot(pos - belt_center, belt_normal) > 0))
        
        upper_stars = sorted([c for c in candidates if c[2]], key=lambda x: x[0][2], reverse=True)
        lower_stars = sorted([c for c in candidates if not c[2]], key=lambda x: x[0][2], reverse=True)
        
        if not (upper_stars and lower_stars):
            return None
        
        betelgeuse = upper_stars[0][0]
        rigel = lower_stars[0][0]
        
        confidence = (1 - (best_belt_score / 30)) * min(1.0, len(candidates) / 5)
        
        return {
            'belt_stars': best_belt['stars'],
            'betelgeuse': (betelgeuse[0], betelgeuse[1]),
            'rigel': (rigel[0], rigel[1]),
            'confidence': confidence
        }

    def identify_orion(self, stars, image_shape):
        """
        Identify Orion constellation from detected stars.
        
        Args:
            stars: List of (x, y, brightness) tuples
            image_shape: Shape of the original image
            
        Returns:
            dict: Identified Orion stars and confidence score
        """
        if len(stars) < 7:  # Minimum stars needed for Orion
            return None
        
        # Convert to normalized coordinates
        height, width = image_shape[:2]
        norm_stars = [(x/width, y/height, b) for x, y, b in stars]
        
        # Find brightest stars that could form Orion's pattern
        candidates = norm_stars[:15]  # Take top 15 brightest stars
        
        # Try to match Orion's pattern
        best_match = None
        best_score = 0
        
        # Function to calculate angle between vectors
        def vector_angle(v1, v2):
            dot = v1[0]*v2[0] + v1[1]*v2[1]
            mags = math.sqrt(v1[0]**2 + v1[1]**2) * math.sqrt(v2[0]**2 + v2[1]**2)
            return math.acos(max(min(dot/mags, 1), -1))
        
        # Try each possible combination of bright stars
        for i, (x1, y1, _) in enumerate(candidates):
            for j, (x2, y2, _) in enumerate(candidates):
                if i == j:
                    continue
                    
                # Check if these could be Betelgeuse and Rigel (shoulder and foot)
                vec1 = (x2-x1, y2-y1)
                vec_len = math.sqrt(vec1[0]**2 + vec1[1]**2)
                
                if 0.3 < vec_len < 0.7:  # Expected relative distance
                    # Look for belt stars
                    potential_belt = []
                    for k, (x3, y3, _) in enumerate(candidates):
                        if k in (i, j):
                            continue
                        
                        # Check if this star could be part of the belt
                        vec2 = (x3-x1, y3-y1)
                        angle = vector_angle(vec1, vec2)
                        
                        if 0.3 < angle < 0.7:  # Expected angle for belt
                            potential_belt.append((x3, y3))
                    
                    # Need at least 3 stars for the belt
                    if len(potential_belt) >= 3:
                        # Calculate match score
                        score = 1.0  # Base score
                        best_match = {
                            'Betelgeuse': (x1*width, y1*height),
                            'Rigel': (x2*width, y2*height),
                            'belt_stars': [(x*width, y*height) for x, y in potential_belt[:3]]
                        }
                        best_score = score
        
        if best_match and best_score > 0.5:
            return best_match
        return None

    def calculate_observer_position(self, image_path):
        """
        Calculate observer's position using Orion as reference.
        """
        try:
            # Detect stars
            img = cv2.imread(image_path)
            detected_stars = self.detect_stars_improved(image_path)
            
            # Try to identify Orion
            #orion_match = self.identify_orion(detected_stars, img.shape)
            orion_match = self.identify_orion_improved(detected_stars, img.shape)
            if not orion_match:
                raise Exception("Could not identify Orion constellation")
            
            # Calculate position based on Orion's orientation
            # Using Betelgeuse and belt stars for orientation
            betelgeuse = np.array(orion_match['Betelgeuse'])
            belt_center = np.mean(orion_match['belt_stars'], axis=0)
            
            # Calculate orientation vector
            orientation = belt_center - betelgeuse
            angle = math.atan2(orientation[1], orientation[0])
            
            # Convert to celestial coordinates
            # This is a simplified calculation for demo purposes
            r = 1.0  # Unit sphere
            latitude = math.pi/2 - angle
            longitude = math.atan2(orientation[1], orientation[0])
            
            x = r * math.cos(latitude) * math.cos(longitude)
            y = r * math.cos(latitude) * math.sin(longitude)
            z = r * math.sin(latitude)
            
            return (x, y, z)
        except Exception as e:
            raise Exception(f"Error calculating position: {str(e)}")

    def visualize_detection(self, image_path, save_path=None):
        """
        Visualize detected stars with improved detection and identification
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise Exception("Failed to load image")
        
        # Use improved detection
        detected_stars = self.detect_stars_improved(image_path)
        
        # Use improved identification
        orion_match = self.identify_orion_improved(detected_stars, img.shape)
        
        # Create visualization overlay
        overlay = img.copy()
        
        # Draw all detected stars
        for x, y, brightness in detected_stars:
            size = int(3 + (brightness/255) * 3)  # Size based on brightness
            cv2.circle(overlay, (int(x), int(y)), size, (0, 255, 0), -1)
        
        if orion_match:
            # Draw belt with thicker lines
            for i in range(len(orion_match['belt_stars'])-1):
                pt1 = tuple(map(int, orion_match['belt_stars'][i]))
                pt2 = tuple(map(int, orion_match['belt_stars'][i+1]))
                cv2.line(overlay, pt1, pt2, (255, 255, 0), 3)
            
            # Draw Betelgeuse and Rigel with larger circles
            bx, by = orion_match['betelgeuse']
            rx, ry = orion_match['rigel']
            
            cv2.circle(overlay, (int(bx), int(by)), 12, (0, 0, 255), 2)
            cv2.circle(overlay, (int(rx), int(ry)), 12, (255, 165, 0), 2)
            
            # Add labels with confidence
            cv2.putText(overlay, f"Betelgeuse", (int(bx)+15, int(by)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(overlay, f"Rigel", (int(rx)+15, int(ry)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
            cv2.putText(overlay, f"Confidence: {orion_match['confidence']:.2f}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        #print("confidence")
        # Blend with original image
        result = cv2.addWeighted(overlay, 0.7, img, 0.3, 0)
        
        if save_path:
            cv2.imwrite(save_path, result)
        return result

def main():
    # Initialize navigator
    navigator = ZenithCelestialNavigator()
    
    # Process image
    image = 'orion_1.jpg'
    image_2 = 'orion_2.png'

    img_2 = cv2.imread(image_2)
    if img_2 is None:
        raise ValueError("Failed to load image. Please check the file path or format.")



    try:
        # Calculate position

        detected_stars = navigator.detect_stars(image_2)
        identify_stars = navigator.identify_orion_improved("orion_2.png", detected_stars, img_2.shape)
        print(identify_stars)
        # x, y, z = navigator.calculate_observer_position(image_2)
        # print(f"Observer position (x, y, z): ({x:.4f}, {y:.4f}, {z:.4f})")
        
        # # Visualize detection
        # navigator.visualize_detection(image_2, 'detected_orion.jpg')
        # print("Visualization saved as 'detected_orion.jpg'")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()