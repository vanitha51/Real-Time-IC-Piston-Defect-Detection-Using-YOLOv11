import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Load YOLO classification model
@st.cache_resource
def load_model(path):
    return YOLO(path)

model = load_model(r"F:\TCS innovent\YOLO 11\Whole Piston\Results\best (1).pt")  # YOLO classification model

st.title("Defect Classification App")
st.write("Upload an image to check if the part is GOOD or DEFECTIVE.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("Classifying...")
    results = model.predict(source=image)

    if not results:
        st.warning("No prediction returned by the model.")
    else:
        pred_class = results[0].names[results[0].probs.top1]
        confidence = results[0].probs.top1conf.item() * 100

        if pred_class == "good_set":
            st.success(f"✅ GOOD PART (Confidence: {confidence:.2f}%)")
        else:
            st.error(f"❌ DEFECT DETECTED (Confidence: {confidence:.2f}%)")
