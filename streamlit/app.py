import time
from io import BytesIO
from operator import mod

import numpy as np
import pandas as pd
import pydicom
from skimage.transform import resize
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img, smart_resize

import streamlit as st


def image_single_to_three_channel(arr):
    """Create 3 channeled imaging

    Args:
        arr (list): Image as list

    Returns:
        image: Converted single channel image to 3 channeled
    """
    # Create 3 channeled numpy array with image dimensions and fill them with zeros
    image = np.zeros((np.array(arr).shape[0], np.array(arr).shape[1], 3))
    # Store same value in each channel
    image[:, :, 0] = arr
    image[:, :, 1] = arr
    image[:, :, 2] = arr

    # Return converted image
    return image


def read_xray(image):
    """Read x-rays on given path

    Args:
        path (string): Path to x-ray
        channels (int): Number of channels

    Returns:
        list: Returns normalized image array with given number of channels
    """
    # Read .dcm image using pydicom library
    dicom = pydicom.dcmread(image)
    # Get numpy array representation
    image = dicom.pixel_array

    reshaped_image = resize(image, (300, 300), anti_aliasing=True)

    # Check if channels need to be added
    image = image_single_to_three_channel(reshaped_image)

    # Return normalized pixel image array
    return np.array([image / 255])


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


if st.button("Submit"):
    xception_model_density = load_model(f"./models/density_xception_model_v40.h5")

    xception_model_under_sampled_cancer = load_model(
        f"./models/under_sampled_cancer_xception_model_v2.h5"
    )

    xception_model_over_sampled_cancer = load_model(
        f"./models/over_sampled_cancer_xception_model_v57.h5"
    )

    X_test = pd.DataFrame(
        [[age, view, laterality]], columns=["age", "view", "laterality"]
    )

    X_test["view"] = X_test["view"].map({"CC": 0, "MLO": 1})
    X_test["laterality"] = X_test["laterality"].map({"Left": 0, "Right": 1})
    image = read_xray(file)

    pred_density = xception_model_density.predict([image, X_test]).flatten()

    pred_density = round(pred_density[0], 2)
    st.write(
        f"Predicted Breast Density: {f'Not Dense with probability ({round((1 - pred_density), 2)})' if round((pred_density)) == 0 else f'Dense with probability ({pred_density})'}"
    )

    X_test["density"] = pred_density

    pred_cancer_under = xception_model_under_sampled_cancer.predict(
        [image, X_test]
    ).flatten()

    pred_cancer_under = round(pred_cancer_under[0], 2)
    st.write(
        f"Predicted Breast Cancer Under Sampled: {f'Cancer Not Found with probability ({round((1 - pred_cancer_under), 2)})' if round((pred_cancer_under)) == 0 else f'Cancer Found with probability ({pred_cancer_under})'}"
    )

    pred_cancer_over = xception_model_over_sampled_cancer.predict(
        [image, X_test]
    ).flatten()

    pred_cancer_over = round(pred_cancer_over[0], 2)
    st.write(
        f"Predicted Breast Cancer Over Sampled: {f'Cancer Not Found with probability ({pred_cancer_over})' if round((pred_cancer_over)) == 0 else f'Cancer Found with probability ({pred_cancer_over})'}"
    )

    st.write("Thank you")

    # show_file = st.empty()

    # content = file.getvalue()

    # if isinstance(file, BytesIO):
    #     show_file.image(file)
    #     test_img = img_to_array(load_img(file))
    #     test_img = smart_resize(test_img, (256, 256))
    #     test_img = np.expand_dims(test_img, axis=0)
    #     print(test_img.shape)

    #     pred = model.predict([test_img])[0]
    #     st.write(
    #         f"The image is: ",
    #         "Not a hotdog" if round(pred[0]) == 1 else "Hotdog",
    #     )
    # else:
    #     df = pd.read_csv(file)
    #     print(df.shape)

    #     st.dataframe(df.head())

if file:
    file.close()
