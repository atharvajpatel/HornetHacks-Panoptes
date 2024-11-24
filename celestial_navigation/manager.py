import star_detection as sd
import star_matching as sm


example_path = "celestial_navigation/example_input.jpg"
image_with_stars, stars = sd.detection(example_path)
sgr = sm.Sagittarius()
sm.draw_constellation_on_image(image_with_stars, stars, sgr)



#ret = sd.get_image()