import star_detection as sd
import star_matching as sm

# solving problems:
# use conda base python interpreter
# deactivate and reactivate
# also just close and open again after installation



# image_with_stars, stars = sd.detection(example_path)
# sgr = sm.Sagittarius()
# sm.draw_constellation_on_image(image_with_stars, stars, sgr)

import cv2
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
import astroquery
from astroquery.astrometry_net import AstrometryNet
from sklearn.neighbors import KDTree


import pandas as pd
from datetime import datetime
import ephem

catalog_path = "hipparcos.csv"


class CelestialNavigator:
    def __init__(self, api_key):
        """
        Initialize the celestial navigation system.
        p
        Args:
            api_key (str): API key for astrometry.net
        """
        self.ast = AstrometryNet()
        self.ast.api_key = api_key
        
        # Load star catalog (Bright Star Catalog)
        self.star_data = self._load_star_catalog()
        self.star_tree = None
        self._build_star_tree()

    def _load_star_catalog(self):
        """Load and process the Yale Bright Star Catalog."""
        # This is a simplified version - in practice, you'd want to load
        # actual star catalog data from a file or database
        # Format: RA (degrees), Dec (degrees), Magnitude
        catalog = pd.read_csv(catalog_path)
        print('hipparcos_data.head()')
        # catalog = pd.DataFrame({
        #     'ra': [],
        #     'dec': [],
        #     'magnitude': []
        # })
        return catalog

    def _build_star_tree(self):
        """Build KD-tree for fast nearest neighbor star matching."""
        if len(self.star_data) > 0:
            coords = np.column_stack((self.star_data['ra'], self.star_data['dec']))
            self.star_tree = KDTree(coords)

    def plate_solve(self, image_path):
        """
        Solve the astrometry of an image using astrometry.net.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            WCS: World Coordinate System solution
        """
        try:
            # Submit the image to astrometry.net
            wcs_header = self.ast.solve_from_image(image_path)
            
            if wcs_header:
                return WCS(wcs_header)
            else:
                raise Exception("Failed to solve image astrometry")
                
        except Exception as e:
            raise Exception(f"Error during plate solving: {str(e)}")

    

    def detection(image_path):
        # Load image
        img = cv2.imread(image_path)

        # Convert to grayscale image
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Set a threshold for brightness (here, set it to 150). originally 
        ret, new = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY)

        # The image is made up of white dots on a black background, so the outlines of the white dots are detected.
        #detected_image, 
        contours, hierarchy = cv2.findContours(new, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # For each contour, find the center of gravity and put it into an array
        stars = []
        for cnt in contours:
            M = cv2.moments(cnt)
            if M['m00'] != 0: # if area is non-zero:
                cx = int(M['m10'] / M['m00']) # m10 is the sum of x pixels in the contour. normalize by area
                cy = int(M['m01'] / M['m00'])
                # centroid coordinates
                stars.append(np.array([cx, cy], dtype='int32'))
            else:
                # If the moment calculation fails, take the first point of the contour
                stars.append(np.array(cnt[0][0], dtype='int32'))

        # Return or print the array of star positions
        #print(stars)

        img_with_contours = img.copy()
        cv2.drawContours(img_with_contours, contours, -1, (0, 255, 0), 2)  # Green contours with thickness 2

        cv2.imshow("Image with Contours", img_with_contours)
        cv2.waitKey(0) # press 0 to exit
        cv2.destroyAllWindows()
        return img_with_contours, stars



    def calculate_position(self, image_path, timestamp=None):
        """
        Calculate observer's position based on the star field image.
        
        Args:
            image_path (str): Path to the celestial image
            timestamp (datetime): Time when the image was taken
            
        Returns:
            tuple: (latitude, longitude) in degrees
        """
        try:
            # If no timestamp provided, use current time
            if timestamp is None:
                timestamp = datetime.utcnow()
            
            # Convert timestamp to astropy Time object
            t = Time(timestamp)
            
            # Solve plate astrometry
            wcs = self.plate_solve(image_path)
            
            # Detect stars in image
            detected_stars = self.detection(image_path)
            
            # Convert pixel coordinates to sky coordinates
            sky_coords = []
            for x, y in detected_stars:
                ra, dec = wcs.pixel_to_world(x, y)
                sky_coords.append(SkyCoord(ra, dec))
            
            # Match detected stars with catalog
            matched_stars = self._match_stars(sky_coords)
            
            # Calculate position using spherical astronomy
            # This is a simplified version - real implementation would use
            # more sophisticated algorithms
            lat, lon = self._calculate_position_from_stars(matched_stars, t)
            
            return lat, lon
            
        except Exception as e:
            raise Exception(f"Error calculating position: {str(e)}")

    def _match_stars(self, detected_coords):
        """Match detected stars with catalog stars."""
        if self.star_tree is None:
            raise Exception("Star catalog not initialized")
            
        matched = []
        for coord in detected_coords:
            # Find nearest catalog star
            dist, idx = self.star_tree.query([[coord.ra.deg, coord.dec.deg]], k=1)
            if dist[0][0] < 1.0:  # Maximum 1 degree separation
                matched.append((coord, self.star_data.iloc[idx[0][0]]))
        return matched

    def _calculate_position_from_stars(self, matched_stars, time):
        """
        Calculate position using matched stars and spherical astronomy.
        This is a simplified implementation - real systems use more
        sophisticated methods.
        """
        # Initialize observer
        observer = ephem.Observer()
        
        # Calculate altitude and azimuth for each matched star
        measurements = []
        for obs_coord, cat_star in matched_stars:
            star = ephem.FixedBody()
            star._ra = obs_coord.ra.rad
            star._dec = obs_coord.dec.rad
            measurements.append((star, cat_star))
        
        # Use multiple star sights to triangulate position
        # This is a simplified example - real implementation would use
        # more sophisticated algorithms like Sumner lines
        lat = 0.0
        lon = 0.0
        
        # Sum weighted contributions from each star
        weight_sum = 0
        for star, cat_data in measurements:
            # Calculate position contribution from this star
            # This is a placeholder for actual astronomical calculations
            star_weight = 1.0 / float(cat_data['mag'])
            lat += star_weight * 0  # Replace with actual calculation
            lon += star_weight * 0  # Replace with actual calculation
            weight_sum += star_weight
        
        if weight_sum > 0:
            lat /= weight_sum
            lon /= weight_sum
            
        return lat, lon

# Example usage
def main():
    # Initialize navigator with your astrometry.net API key
    navigator = CelestialNavigator('')
    
    # Calculate position from an image
    #image_path = 'path_to_your_star_field_image.jpg'
    example_path = "celestial_navigation/example_input.jpg"
    try:
        latitude, longitude = navigator.calculate_position(example_path)
        print(f"Estimated position: {latitude:.4f}°N, {longitude:.4f}°E")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()