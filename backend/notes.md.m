
# Lets start with a target image of a component and an empty html file.
# Assume for now that the target image has the dimensions of the view port and a single button at the center
# The goal is for this program to start with an empty html file and correctly construct a button that is pixel perfect 
# relative to the target image.
# So the program will be driven through a loop of compare --> modify --> compare --> modify
# Compare should use image comparison to compare the target and the current html
# Keep track of the current similarity score of the button and the target, exit the program when it converges 
# under some set threshold

# ideally, this thing could recursively segment areas of difference, but that is a whole can of worms



# Granular pixel perfect matching
# Get computed font size, styling from the current HTML
# Segment the button so that the internal text is isolated
# Compare heights and determine the best font size and styling that fits.

# Apply this for color, background color, 