import time
from io import BytesIO
from operator import mod

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img, smart_resize

import streamlit as st


def calculate_time_took(start, progress_bar):
    processed_time = time.time()
    time_elapsed = round(processed_time - start_time, 4)
    st.info(f"Time Elapsed {time_elapsed}")


# model = load_model("models\*********.h5")
# current datetime
start_time = time.time()
my_progress_bar = st.progress(0)


st.title("Cancer Classification Model")

st.subheader("Upload DICOM File Type Mammogram")


file = st.file_uploader("Upload image format: .dcm")


if file and file.name.split(".")[1] != "dcm":
    st.error(
        f"Given file is not dicom file type, given file type is '{file.name.split('.')[1]}'",
        icon="🚨",
    )
elif file and file.name.split(".")[1] == "dcm":
    st.info(
        "Thank you for providing dicom file, I will preprocess the mammogram and run prediction model"
    )


laterality = st.radio("Pick a laterality of breast", ["Left", "Right"])
st.info(laterality)

view = st.radio("Pick a view of breast", ["CC", "MLO"])
st.info(view)


age = st.slider("Provide age of the patient", 0, 130, 25)
st.info(age)


# if st.button("Submit"):
#     show_file = st.empty()

#     content = file.getvalue()

#     if isinstance(file, BytesIO):
#         show_file.image(file)
#         test_img = img_to_array(load_img(file))
#         test_img = smart_resize(test_img, (256, 256))
#         test_img = np.expand_dims(test_img, axis=0)
#         print(test_img.shape)

#         pred = model.predict([test_img])[0]
#         st.write(
#             f"The image is: ",
#             "Not a hotdog" if round(pred[0]) == 1 else "Hotdog",
#         )
#     else:
#         df = pd.read_csv(file)
#         print(df.shape)

#         st.dataframe(df.head())

if file:
    file.close()

calculate_time_took(start_time, my_progress_bar)
