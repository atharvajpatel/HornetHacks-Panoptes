# Match Detected Stars to a Star Catalog
import cv2
import numpy as np

class Sagittarius:
    def __init__(self):
        self.SGR5 = {"ANGS":[-48.5], "D":[1.237],
                     "JCT":[], "BP":[], "REST":[]
                    }
        self.SGR4 = {"ANGS":[+123.1, +27.85, -37.55],
                     "D":[0.8143, 0.9881, 0.59727],
                     "JCT":[0], "BP":[], "REST":[self.SGR5]
                    }
        self.SGR3 = {"ANGS":[-128, +15.17],
                     "D":[0.7564, 2.0678],
                     "JCT":[], "BP":[], "REST":[]
                    }
        self.SGR2 = {"ANGS":[-65.5, +96.3],#64.5
                     "D":[0.8987, 0.3662],
                     "JCT":[0], "BP":[], "REST":[self.SGR3]
                    }
        self.SGR = {"ANGS":[+85, +47, -37, -29],
                    "D":[1.25, 0.724, 1.934, 1.354],
                    "JCT":[0, 2], "BP":[], "REST":[self.SGR2, self.SGR4],
                    "N":4, "MAX": 13 #13
                   }
        self.iau5 = {"ANGS":[None], "D":[7]}
        self.iau4 = {"ANGS":[None], "D":[5]}
        self.iau3 = {"ANGS":[None], "D":[7]}
        self.iau2 = {"ANGS":[None], "D":[3]}
        self.iau = {"ANGS":[+85, +47, -37, +93, -58, +124],
                    "D":[1.25, 0.724, 1.934, 1.093, 0.731, 1.625],
                    "JCT":[-2, -2, 1, 3], "BP":[],
                    "REST":[self.iau2, self.iau3, self.iau4, self.iau5],
                    "N":6, "MAX": 8 # 本当は7
                   }
        self.line = self.SGR
        self.ja_name = "いて座"
        self.en_name = "Sagittarius"
        self.short_name = "SGR"
sgr = Sagittarius()


def draw_constellation_on_image(image, stars, constellation):
    """
    Draws the Sagittarius constellation on top of the provided image.

    Parameters:
    image: The original image on which the constellation will be drawn.
    stars: List of [x, y] coordinates for detected star positions.
    constellation: A dictionary defining the constellation structure (e.g., angles, distances, and relationships).
    """
    # Clone the image to avoid modifying the original
    output_image = image.copy()

    # Scale factor for distances
    scale = 200  # Adjust based on your image dimensions

    # Start with the main constellation (line)
    center_star_index = 0  # Assume the main star is the first in the `stars` list
    center_star = stars[center_star_index]

    # Draw the main star
    cv2.circle(output_image, tuple(center_star), 5, (0, 255, 0), -1)  # Main star in green

    # Iterate through the angles and distances in the constellation
    for i, angle in enumerate(constellation.line["ANGS"]):
        if angle is None:
            continue

        # Calculate the position of the connected star
        distance = constellation.line["D"][i] * scale
        dx = int(distance * np.cos(np.radians(angle)))
        dy = int(distance * np.sin(np.radians(angle)))
        connected_star = [center_star[0] + dx, center_star[1] - dy]  # Adjust for OpenCV's y-axis orientation

        # Draw the connection line
        cv2.line(output_image, tuple(center_star), tuple(connected_star), (255, 255, 255), 1)  # White lines
        # Draw the connected star
        cv2.circle(output_image, tuple(connected_star), 5, (0, 0, 255), -1)  # Connected stars in red

    # Display the image with the constellation drawn
    cv2.imshow("Sagittarius Constellation", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
