import cv2
from PIL import Image

import os
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import pandas as pd
import numpy as np
from helpers import Homography, PitchImage
from pitch import FootballPitch
import xml.etree.cElementTree as ET
from xml.dom import minidom

area_options_dic = {'Offensive Half': [[525, 0], [262.5, 0], [262.5, 340], [525, 340]],
                    'Defensive Half': [[262.5, 0], [0, 0], [0, 340], [262.5, 340]],
                    'Offensive Middle Third': [[442.5, 0], [262.5, 0], [262.5, 340], [442.5, 340]],
                    'Defensive Middle Third': [[262.5, 0], [82.5, 0], [82.5, 340], [262.5, 340]],
                    'Offensive Penalty Area': [[525, 69.2], [442.5, 69.2], [442.5, 270.8], [525, 270.8]],
                    'Defensive Penalty Area': [[82.5, 69.2], [0, 69.2], [0, 270.8], [82.5, 270.8]],
                    'Offensive Half Left': [[525, 0], [262.5, 0], [262.5, 170], [525, 170]],
                    'Offensive Half Right': [[525, 170], [262.5, 170], [262.5, 340], [525, 340]],
                    'Defensive Half Left': [[262.5, 0], [0, 0], [0, 170], [262.5, 170]],
                    'Defensive Half Right': [[262.5, 170], [0, 170], [0, 340], [262.5, 340]],
                    'Offensive Middle Third Left': [[442.5, 0], [262.5, 0], [262.5, 170], [442.5, 170]],
                    'Offensive Middle Third Right': [[442.5, 170], [262.5, 170], [262.5, 340], [442.5, 340]],
                    'Defensive Middle Third Left': [[262.5, 0], [82.5, 0], [82.5, 170], [262.5, 170]],
                    'Defensive Middle Third Right': [[262.5, 170], [82.5, 170], [82.5, 340], [262.5, 340]],
                    'Offensive Penalty Area Left': [[525, 69.2], [442.5, 69.2], [442.5, 170], [525, 170]],
                    'Offensive Penalty Area Right': [[525, 170], [442.5, 170], [442.5, 270.8], [525, 270.8]],
                    'Defensive Penalty Area Left': [[82.5, 69.2], [0, 69.2], [0, 170], [82.5, 170]],
                    'Defensive Penalty Area Right': [[82.5, 170], [0, 170], [0, 270.8], [82.5, 270.8]]}

area_options_norm_dic = {'Offensive Half': [[.5, -.5], [0, -.5], [0, .5], [.5, .5]],
                         'Defensive Half': [[0, -.5], [-.5, -.5], [-.5, .5], [0, .5]],
                         'Offensive Middle Third': [[.343, -.5], [0, -.5], [0, .5], [.343, .5]],
                         'Defensive Middle Third': [[0, -.5], [-.343, -.5], [-.343, .5], [0, .5]],
                         'Offensive Penalty Area': [[.5, -.296], [.343, -.296], [.343, .296], [.5, .296]],
                         'Defensive Penalty Area': [[-.343, -.296], [-.5, -.296], [-.5, .296], [-.343, .296]],
                         'Offensive Half Left': [[.5, -.5], [0, -.5], [0, 0], [.5, 0]],
                         'Offensive Half Right': [[.5, 0], [0, 0], [0, .5], [.5, .5]],
                         'Defensive Half Left': [[0, -.5], [-.5, -.5], [-.5, 0], [0, 0]],
                         'Defensive Half Right': [[0, 0], [-.5, 0], [-.5, .5], [0, .5]],
                         'Offensive Middle Third Left': [[.343, -.5], [0, -.5], [0, 0], [.343, 0]],
                         'Offensive Middle Third Right': [[.343, 0], [0, 0], [0, .5], [.343, .5]],
                         'Defensive Middle Third Left': [[0, -.5], [-.343, -.5], [-.343, 0], [0, 0]],
                         'Defensive Middle Third Right': [[0, 0], [-.343, 0], [-.343, .5], [0, .5]],
                         'Offensive Penalty Area Left': [[.5, -.296], [.343, -.296], [.343, 0], [.5, 0]],
                         'Offensive Penalty Area Right': [[.5, 0], [.343, 0], [.343, .296], [.5, .296]],
                         'Defensive Penalty Area Left': [[-.343, -.296], [-.5, -.296], [-.5, 0], [-.343, 0]],
                         'Defensive Penalty Area Right': [[-.343, 0], [-.5, 0], [-.5, .296], [-.343, .296]]}

area_options_320_dic = {k: ((np.array(v)+0.5)*319).tolist() for k,v in area_options_norm_dic.items()}

key_pts_dic = {
    '0': [3, 3], # top left
    '1': [3, 66],
    '2': [51, 65],
    '3': [3, 117],
    '4': [17, 117],
    '5': [3, 203],
    '6': [17, 203],
    '7': [3, 255],
    '8': [51, 254], # bottom left
    '9': [3, 317],
    '10': [160, 3],
    '11': [160, 160],
    '12': [160, 317],
    '13': [317, 3], # top right
    '14': [317, 66],
    '15': [270, 66],
    '16': [317, 118],
    '17': [304, 118],
    '18': [317, 203],
    '19': [304, 203],
    '20': [317, 255],
    '21': [270, 255],
    '22': [317, 317], # bottom right
    '23': [51, 128],
    '24': [51, 193],
    '25': [161, 118],
    '26': [161, 203],
    '27': [270, 128],
    '28': [270, 192],
}

st.set_option('deprecation.showfileUploaderEncoding', False)
st.set_page_config(page_title='BirdsPyView', layout='wide')
st.title('Narya - Manual Labeling - Pitch')
pitch = FootballPitch()

path = os.path.expanduser('homography/raw_images/')
files = os.listdir(path)
files = [x for x in files if x != '.DS_Store']
file = files[0]
uploaded_file = cv2.imread(os.path.join(path, file))
    
image = PitchImage(pitch, image=uploaded_file, border = 100)

st.title('Pitch Markers')

lines_expander = st.beta_expander('In the dropdown, choose the largest pitch area you are comfortably seeing in the image. \
                                  Then mark the four corners starting on the top left, proceeding counter-clockwise. You may mark a point on the border outside the image.',
                                  expanded=True)
with lines_expander:
    col1, col2, col3 = st.beta_columns([2,0.5,1])

    with col1:
        canvas_image = st_canvas(
            fill_color = "rgba(255, 165, 0, 0.3)", 
            stroke_width = 2,
            stroke_color = '#000000',
            background_image = image.im,
            width = image.im.width,
            height = image.im.height,
            drawing_mode = "circle",
            key = "canvas",
        )

    with col2:
        store = st.button('Store Results')
        delete = st.button('Delete Image')

    with col3:
        area_selection = st.selectbox('What do you see', list(area_options_dic), key='area_select', index=0)
        if area_selection:
            st.image('images/' + '_'.join(area_selection.split()) + '.png', use_column_width=True)

if canvas_image.json_data is not None:
    num_markers = len(canvas_image.json_data["objects"])
    markers = pd.json_normalize(canvas_image.json_data["objects"])
    
    if num_markers >= 4:
        
        pts_dst = np.array(area_options_dic[area_selection])
        pts_dst_norm = np.array(area_options_norm_dic[area_selection])
        pts_dst_320 = np.array(area_options_320_dic[area_selection])
        
        pts_left = [x - 100 for x in markers['left'].tolist()] # Remove border
        pts_top = [x - 100 for x in markers['top'].tolist()] # Remove border
        
        pts_left_320 = [x/511*319 for x in pts_left] # Normalize
        pts_top_320 = [x/511*319 for x in pts_top] # Normalize
        
        pts_left_norm = [(x - 511/2)/511 for x in pts_left] # Normalize
        pts_top_norm = [(x - 511/2)/511 for x in pts_top] # Normalize
        
        pts_src = list(zip(pts_left, pts_top))
        pts_src_norm  = list(zip(pts_left_norm, pts_top_norm))
        pts_src_320 = list(zip(pts_left_320, pts_top_320))
        
        h = Homography(pts_src, pts_dst)
        h_norm, _ = cv2.findHomography(np.array(pts_src_norm), np.array(pts_dst_norm))
        h_320, _ = cv2.findHomography(np.array(pts_src_320), np.array(pts_dst_320))
        
        h_out, conv_img = h.apply_to_image(uploaded_file)
        conv_img = Image.fromarray(conv_img)

        key_points_trans = {k: cv2.perspectiveTransform(np.array([[v, v]], dtype=np.float32), np.linalg.pinv(h_320))[0][0].tolist() for k,v in key_pts_dic.items()}
        key_points_trans = {k: v for k,v in key_points_trans.items() if (0 <= v[0] <= 319) and (0 <= v[1] <= 319)}
        
        pnt_img = cv2.resize(uploaded_file, (320,320))
        
        for key, pnt in key_points_trans.items():
        
            pnt_img = cv2.circle(pnt_img, tuple(np.array(pnt, dtype=np.float32)), radius=1, color=(0, 0, 255), thickness=-1)
        
        with lines_expander:
            col1, col_, col2 = st.beta_columns([2,0.5,2])
            with col1:
                st.write('Homography')
                st.image(conv_img)
            with col2:
                st.write('Keypoints')
                st.image(Image.fromarray(pnt_img))

        if store:
            os.remove(os.path.join(path, file))
            key = len(os.listdir('homography/homography/JPEGImages/'))
            cv2.imwrite('homography/homography/JPEGImages/frame_%d.jpg' % key, uploaded_file)
            with open('homography/homography/homographies/frame_%d.npy' % key, 'wb') as f:
                 np.save(f, h_norm)
                
            annotation = ET.Element("annotation")
            ET.SubElement(annotation, "folder").text = "JPEGImages"
            ET.SubElement(annotation, "filename").text = str(key) + ".jpg"
            size = ET.SubElement(annotation, "size")
            ET.SubElement(size, "width").text = "320"
            ET.SubElement(size, "height").text = "320"
            ET.SubElement(size, "depth").text = "3"
            
            for name, pnt in key_points_trans.items(): 
                object = ET.SubElement(annotation, "object")
                ET.SubElement(object, "name").text = name
                ET.SubElement(object, "difficult").text = "0"
                keypoints = ET.SubElement(object, "keypoints")
                ET.SubElement(keypoints, "x1").text = str(int(pnt[0]))
                ET.SubElement(keypoints, "y1").text = str(int(pnt[1]))
                ET.SubElement(keypoints, "v1").text = "2"
                bndbox = ET.SubElement(object, "bndbox")
                ET.SubElement(bndbox, "xmin").text = str(int(pnt[0]))
                ET.SubElement(bndbox, "ymin").text = str(int(pnt[1]))
                ET.SubElement(bndbox, "xmax").text = str(int(pnt[0]))
                ET.SubElement(bndbox, "ymax").text = str(int(pnt[1]))

            xmlstr = minidom.parseString(ET.tostring(annotation)).toprettyxml(indent="   ")
            
            with open('homography/keypoints/annotations/frame_%d.xml' % key, 'w') as f:
                f.write(xmlstr)
            
            cv2.imwrite('homography/keypoints/JPEGImages/frame_%d.jpg' % key, cv2.resize(uploaded_file, (320,320)))
     
            
if delete:
    os.remove(os.path.join(path, file))            