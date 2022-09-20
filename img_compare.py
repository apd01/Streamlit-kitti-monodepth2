import streamlit as st
from streamlit_image_comparison import image_comparison
import cv2

from augmentations import augmentation_choices, transform_image
import get_monodepth
import os

st.set_page_config("Monodepth vs Ground Truth", "***", layout="wide")


# st.image(
#     "https://eak2mvmpt4a.exactdn.com/wp-content/uploads/2020/07/A-Guide-to-Lidar-Wavelengths-Velodyne-Lidar-AlphaPrime-1024x508.jpg",
#     width=1024,
# )

st.header("Depth Prediction")
st.write("")
"This app applies various image augmentation techniques to images in the KITTI dataset and calculates the resulting depth maps."
st.write("")

img_path = r'./images/0'
img_suffix = '.png'

placeholder_image = st.image(
    img_path+img_suffix,
    width=1366,
)
img_augmented_path = ''
selectbox_state = ''

model_types = [ "mono_640x192",
                "stereo_640x192",
                "mono+stereo_640x192",
                "mono_no_pt_640x192",
                "stereo_no_pt_640x192",
                "mono+stereo_no_pt_640x192",
                "mono_1024x320",
                "stereo_1024x320",
                "mono+stereo_1024x320" ]

with st.sidebar.form(key="selectAugmentation"):
    get_monodepth.download()
    selectbox_state = st.selectbox("Choose an augmentation type", augmentation_choices)
    numberinput_threshold = st.number_input(
        """Set variance for Gaussian augmentation""",
        value=0.05,
        min_value=0.,
        max_value=1.,
        step=0.01,
        format="%0.2f",
    )
    st.markdown(
        # '<p class="italic">Settings are currently placeholders</p>', # small-font
        '<i>^ These settings are currently placeholders</i>', # small-font
        unsafe_allow_html=True,
    )
    generate_augmentation_pressed = st.form_submit_button("Generate augmentation")
    
# Sidebar to choose depth prediction model
with st.sidebar.form(key="selectModel"):
    selectbox_model = st.selectbox("Choose a model", ["DepthFormer " + a for a in model_types])
    
    # print(["DepthFormer " + a for a in model_types][0][12:])
    
    build_depth_map_pressed = st.form_submit_button("Build Depth Map")
# 
if generate_augmentation_pressed:
    # print("Generate augmentation")
    img_path = r'./images/0'
    img_suffix = '.png'
    img = transform_image(selectbox_state, cv2.imread(img_path+img_suffix))
    img_augmented_path = img_path+'_'+selectbox_state+img_suffix
    cv2.imwrite(img_augmented_path, img)

    print('img_augmented_path: ' + img_augmented_path)
    # breakpoint()

    st.markdown("### Augmented Image")
    placeholder_image.empty()
    placeholder_image = image_comparison(
        width = 1366,
        img1=img_path+img_suffix,
        img2=img_augmented_path,
        label1="Original",
        label2="Augmented - " + selectbox_state,
    )

if build_depth_map_pressed:
    print("--- Build depth map ---")
    placeholder_image.empty()  
    img_path = r'./images/0'
    img_suffix = '.png'
    img_augmented_path = img_path+'_'+selectbox_state+img_suffix
    
    with st.spinner(text="Generating original depth map..."):
        monodepth_command = r'python .\monodepth2\test_simple.py --image_path' + r' ' +  img_augmented_path + r' --model_name ' + selectbox_model[12:] + r' --pred_metric_depth'
        print("monodepth_command: " + monodepth_command)
        os.system(monodepth_command)
        
        # TODO: if it doesn't already exist...
    with st.spinner(text="Generating augmented depth map..."):
        # monodepth_command = r'python .\monodepth2\test_simple.py --image_path' + r' ' +  img_path+img_suffix + r' --model_name mono+stereo_640x192 --pred_metric_depth'
        monodepth_command = r'python .\monodepth2\test_simple.py --image_path' + r' ' +  img_path+img_suffix + r' --model_name ' + selectbox_model[12:] + r' --pred_metric_depth'
        os.system(monodepth_command)

    placeholder_image = image_comparison(
        width = 1366,
        img1=img_augmented_path,
        img2=img_path+r'_' + selectbox_state + r'_disp.jpeg',
        label1="Augmented - " + selectbox_state,
        label2="Depth map - " + selectbox_state,
    )
    
    placeholder_image = image_comparison(
        width = 1366,
        img1=img_path+r'_disp.jpeg',
        img2=img_path+r'_' + selectbox_state + r'_disp.jpeg',
        label1="Depth map - original",
        label2="Depth map - " + selectbox_state,
    )