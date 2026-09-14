This project is designed to detect the direction of direct arrows using Image Processing techniques.
The image is first converted to Grayscale, and Gaussian Blur is applied to reduce noise. Then, Otsu Thresholding is used to separate the arrow from the background.
Next, Contours are extracted to identify the arrows in the image. For each detected arrow, the system calculates the Centroid and identifies the Arrow Tip using contour approximation and angle analysis.
Finally, the position of the Arrow Tip is compared with the Centroid to classify the arrow direction as:
UP, DOWN, LEFT, or RIGHT.
