This is a Streamlit app heavily based on [BirdsPyView](https://github.com/rjtavares/BirdsPyView) for efficiently generating training lables for the [narya API](https://github.com/DonsetPG/narya).

# Installation

First make sure you have **Python** installed.

Then install **OpenCV**, **Streamlit** and **Streamlit Drawable Canvas**:

    pip install opencv-python
    pip install streamlit
    pip install streamlit-drawable-canvas

Finally, clone the repo and inside narya-label-creator run:

    streamlit run label_vertical_pitch.py or
    
    streamlit run label_player.py

# Getting started

## Labeling pitch for homography estimation

1. Make sure that homography/raw_images/ contains images you would like to label (dimension 512x512).

2. Open the tool as outline above.

3. Pick the largest area of the pitch you are comfortably seeing (e.g. offensive half, penalty area, ...) in the dropdown menu on the right.

4. Draw four markers by clicking on the canvas (drawing on the border is possible) starting at the top left to then proceed counter-clockwise.

5. After the fourth marker is provided the app will generate the homography estimation and the full keypoints mask.

6. If you are happy with the results click 'Store Results'. The results and the original image will be stored in the homography and keypoints folder. The original image will be removed from raw_images.

7. To get the next image to label, simply refresh the page.

## Labeling players with bounding boxes

1. Make sure that player_tracking/raw_images/ contains images you would like to label (dimension 1024x1024).

2. Open the tool as outline above.

3. Draw boxes around every player and referee you see on the pitch.

4. When you are happy click 'Accept Data'. The coordinates of the bounding boxes will be aggregatd into a data frame that appears on the right.

5. If you are happy with the data click 'Store Results'. The results and the original image will be stored in the respective folders. The original image will be removed from raw_images.

6. If you encounter an image you don't want to label click 'Delete Image'. The image will be removed from raw_images.

7. To get the next image to label, simply refresh the page.

# Demo

![](pitch_demo.gif?raw=true)