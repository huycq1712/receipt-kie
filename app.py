import streamlit as st
from PIL import Image
import numpy as np
import time
from main import Pipeline

import time

st.set_page_config(page_title="INVOICE INFORMATION EXTRACTION", page_icon=":smiley:", layout="wide", initial_sidebar_state="expanded")
st.title("INVOICE INFORMATION EXTRACTION")


from background_remove import background_remove
from image_rotation import image_rotation
from text_detection import text_detection
from text_recognition import text_recognition
from pick import key_information_extractor

import numpy as np
import cv2
import PIL
from PIL import Image, ImageDraw, ImageFont

if __name__ == '__main__':
    pass

@st.experimental_singleton
def load_model():
    return Pipeline()


def main():
    img_file_buffer = st.file_uploader("Upload a file", type=["pdf", "png", "jpg", "jpeg"])
    pipeline = load_model()
    if img_file_buffer is not None:
        image_ori = Image.open(img_file_buffer)

        if st.button('Extract Information'):
            with st.spinner('Wait for it...'):
                image_br, image_final, image_text_box, result = pipeline.run(image_ori)
                image_result = [image_ori ,image_br, image_final, image_text_box]

            print(result)

            st.header("Result")
            st.table(result)
            layout = st.columns(4)
            
            header_image = ["Origin Image.", "Background Remove.", "Aligned Image.", "Text detection."]
            for i in range(len(header_image)):
                with layout[i]:
                    st.header(header_image[i])
                    st.image(image_result[i], use_column_width=False, width=300)
    

if __name__ == '__main__':
    
    main()