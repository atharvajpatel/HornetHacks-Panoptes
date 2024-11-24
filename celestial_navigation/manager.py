import star_detection as sd
import star_matching as sm


example_path = "celestial_navigation/example_input.jpg"
stars = sd.detection(example_path)
sgr = sm.Sagittarius()
sm.draw_constellation(stars, sgr)



#ret = sd.get_image()