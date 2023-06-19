from PIL import Image, ImageDraw
from flask import Flask, render_template, request, redirect, send_file
import os
import numpy as np


def average_colour(image):
    # conversion of image to numpy array where each pixel rgb values will be stored as element of the array
    image_arr = np.asarray(image)

    # get average rgb colour of the image quadrant after every recursion
    avg_color_per_row = np.average(image_arr, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)

    avg_color = np.nan_to_num(avg_color)

    return int(avg_color[0]), int(avg_color[1]), int(avg_color[2])


def weighted_average(hist):
    total = sum(hist)

    if total > 0:
        mean = sum(r * i for i, r in enumerate(hist)) / total
        variance = sum(r * (i - mean) ** 2 for i, r in enumerate(hist)) / total
        sd = variance ** 0.5
    else:
        sd = 0

    return sd


def get_detail(hist):
    red_detail = weighted_average(hist[:256])
    green_detail = weighted_average(hist[256:512])
    blue_detail = weighted_average(hist[512:768])

    # combining the channel intensities into a single luminance histogram
    detail_intensity = red_detail * 0.2989 + green_detail * 0.5870 + blue_detail * 0.1141

    return detail_intensity


class Quadrant:
    def __init__(self, image, bbox, depth):
        self.bbox = bbox
        self.depth = depth
        self.children = None
        self.leaf = False

        # image should be crop according to the size of the quadrant in each Quadrant() call
        image = image.crop(bbox)
        # this would get us the count of the pixels of intensities r, g, b of the quadrant which ranges from 0 to 255 for each color
        hist = image.histogram()

        self.colour = average_colour(image)  # this will be used as avg rgb color of the quadrant
        self.detail = get_detail(hist)

    def split_quadrant(self, image):
        left, top, right, bottom = self.bbox

        # middle lines for each quadrant
        x_middle = left + (right - left) / 2
        y_middle = top + (bottom - top) / 2

        # splitting the Quadrants into four other quadrants
        u_left = Quadrant(image, (left, top, x_middle, y_middle), self.depth + 1)
        u_right = Quadrant(image, (x_middle, top, right, y_middle), self.depth + 1)
        d_left = Quadrant(image, (left, y_middle, x_middle, bottom), self.depth + 1)
        d_right = Quadrant(image, (x_middle, y_middle, right, bottom), self.depth + 1)

        # adding these four children qudrants(nodes) to the parent Quadrants's children attribute
        self.children = [u_left, u_right, d_left, d_right]


class Quadtree:

    def __init__(self, image, depth, detail):
        self.width, self.height = image.size

        self.max_depth = 0

        self.threshold_depth = depth
        self.threshold_detail = detail

        # start of compression
        self.start(image)

    def start(self, image):

        # the getbbox() method is used to obtain the (x1, y1, x2, y2) boundaries of the image
        self.root = Quadrant(image, image.getbbox(), 0)

        self.build(self.root, image)

    def build(self, quadrant, image):
        if quadrant.depth >= self.threshold_depth or quadrant.detail <= self.threshold_detail:
            if quadrant.depth > self.max_depth:
                self.max_depth = quadrant.depth

            quadrant.leaf = True
            return

        quadrant.split_quadrant(image)

        for children in quadrant.children:
            self.build(children, image)

    def create_image(self, depth):

        # this creates an empty canvas with the same dimensions of the image
        image = Image.new('RGB', (self.width, self.height))
        draw = ImageDraw.Draw(image)

        # for drawing the outlines for the image
        draw.rectangle((0, 0, self.width, self.height), (0, 0, 0))

        leaf_quadrants = self.get_leaf_quadrants(depth)

        # drawing the rectangles for each quadrant for each leaf quadrant
        for quadrant in leaf_quadrants:
            draw.rectangle(quadrant.bbox, quadrant.colour)

        return image

    def get_leaf_quadrants(self, depth):

        quadrants = []

        self.recursive_search(self.root, depth, quadrants.append)

        return quadrants

    def recursive_search(self, quadrant, max_depth, append_leaf):
        if quadrant.leaf == True or quadrant.depth == max_depth:
            append_leaf(quadrant)
        elif quadrant.children != None:
            for child in quadrant.children:
                self.recursive_search(child, max_depth, append_leaf)


#driver code

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DOWNLOAD_FOLDER'] = 'downloads'


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/compress', methods=['POST'])
def compress():
    # Get the uploaded file
    uploaded_file = request.files['image']

    # Save the uploaded file to the upload folder
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
    uploaded_file.save(file_path)

    # Process the uploaded image and create the quadtree
    image = Image.open(file_path)

    # Get the compression level from the form
    compression_level = request.form['compression_level']
    depth, detail = get_compression_parameters(compression_level)

    quadtree = Quadtree(image, depth, detail)  # Set the desired depth for the quadtree

    # Create the compressed image
    compressed_image = quadtree.create_image(depth - 1)

    # Save the compressed image to the download folder
    download_path = os.path.join(app.config['DOWNLOAD_FOLDER'], 'compressed_image.jpg')
    compressed_image.save(download_path)

    # Redirect to the download page
    return render_template('index.html', compressed_image='/download')



@app.route('/download')
def download():
    # Get the path to the compressed image
    download_path = os.path.join(app.config['DOWNLOAD_FOLDER'], 'compressed_image.jpg')

    # Send the file for download
    return send_file(download_path, as_attachment=True)


def get_compression_parameters(compression_level):
    if compression_level == 'low':
        return 9, 9
    elif compression_level == 'medium':
        return 9, 10
    elif compression_level == 'high':
        return 9, 1


if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(app.config['DOWNLOAD_FOLDER']):
        os.makedirs(app.config['DOWNLOAD_FOLDER'])

    app.run(debug=True)