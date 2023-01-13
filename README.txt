FINAL PROJECT
Various Image Filters, Winslow Church, Fall 2022

To test the filters on your own image, upload the picture to images
and then change the image_path to the new picture

CARTOON FILTER
-------------------
I built off of the kmeans function that we have already implemented
from homework 6. I overlayed edge detection with a softened original
image to give a cartoon effect.

COLOR BLOCK FILTER
-------------------
With bilateral filtering, I merged the major colors of the original
image to create a more blocky result. Each pixel takes into account 
its neighboring pixels to create the final image. Then just by messing
with the colors a bit we can make it look like a color pop filter too!

SEPIA FILTER
-------------------
This old-timey filter was implemented in a similar way as the 
dim_image function from a previous homework. In this case, however, 
I also had to edit the RGB values to alter the overall color. The 
specific formula for the red, green, and blue values I found online.
