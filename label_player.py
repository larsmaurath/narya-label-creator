import cv2
import os
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import pandas as pd
from helpers import PitchImage
from pitch import FootballPitch
import xml.etree.cElementTree as ET
from xml.dom import minidom

st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_page_config(page_title='BirdsPyView', layout='wide')
st.title('Narya - Manual Labeling - Player Tracking')
pitch = FootballPitch()

path = os.path.expanduser('player_tracking/raw_images/')
files = os.listdir(path)
files = [x for x in files if x != '.DS_Store']
file = files[0]
uploaded_file = cv2.imread(os.path.join(path, file))
    
image = PitchImage(pitch, image=uploaded_file)

st.write('Draw rectangles over players (and referees) on the image. '+
         'The player location is assumed to be the middle of the base of the rectangle. ')
st.write('Once you are happy press "Accept Data", double check coordinates in the dataframe and press "Store Results" to store down.')
st.write('If the image is not usable you can delete it with "Delete Image". To get the next image simply refresh the page.')

p_col1, p_col2, p_col_, p_col3 = st.beta_columns([2,1,0.5,1])

with p_col2:
    object_to_label = st.selectbox('Player or Ball?', ('Player', 'Ball'))
    update = st.button('Accept Data')
    store = st.button('Store Results')
    delete = st.button('Delete Image')
    original = True #st.checkbox('Select on original image', value=True)

image2 = image.get_image(original)
height2 = image2.height
width2 = image2.width
stroke_color = '#0000FF' if object_to_label == 'Ball' else '#000000'

with p_col1:
    canvas_converted = st_canvas(
        fill_color = 'rgba(255, 165, 0, 0.3)',
        stroke_width = 2,
        stroke_color = stroke_color,
        background_image = image2,
        drawing_mode = 'rect',
        update_streamlit = update,
        height = height2,
        width = width2,
        key='canvas2',
    )

if canvas_converted.json_data is not None:
    if len(canvas_converted.json_data['objects'])>0:
        dfCoords = pd.json_normalize(canvas_converted.json_data['objects'])
        
        if original:
            dfCoords['xmin'] = dfCoords['left']
            dfCoords['xmax'] = dfCoords['left'] + dfCoords['width']
            dfCoords['ymin'] = dfCoords['top']
            dfCoords['ymax'] = dfCoords['top'] + dfCoords['height']
            dfCoords['entity'] = ['Ball' if x == '#0000FF' else 'Player' for x in dfCoords['stroke']]

    with p_col3:
        st.write('Player Coordinates:')
        st.dataframe(dfCoords[['xmin', 'xmax', 'ymin', 'ymax', 'entity']])
        
if store:
    os.remove(os.path.join(path, file))
    key = len(os.listdir('player_tracking/Annotations/'))       
        
    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = 'JPEGImages'
    ET.SubElement(annotation, 'filename').text = str(key) + '.jpg'
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = '1024'
    ET.SubElement(size, 'height').text = '1024'
    ET.SubElement(size, 'depth').text = '3'
    
    for index, row in dfCoords.iterrows(): 
        object = ET.SubElement(annotation, 'object')
        ET.SubElement(object, 'name').text = row['entity']
        ET.SubElement(object, "difficult").text = "0"
        bndbox = ET.SubElement(object, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(int(row['xmin'])*2)
        ET.SubElement(bndbox, 'ymin').text = str(int(row['ymin'])*2)
        ET.SubElement(bndbox, 'xmax').text = str(int(row['xmax'])*2)
        ET.SubElement(bndbox, 'ymax').text = str(int(row['ymax'])*2)

    xmlstr = minidom.parseString(ET.tostring(annotation)).toprettyxml(indent='   ')
    
    with open('player_tracking/Annotations/frame_%d.xml' % key, 'w') as f:
        f.write(xmlstr)
    
    cv2.imwrite('player_tracking/JPEGImages/frame_%d.jpg' % key, uploaded_file)

if delete:
    os.remove(os.path.join(path, file))