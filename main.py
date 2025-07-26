import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

model = load_model("rmodel.keras", compile=False)

st.title("Real Or Fake Face")

uploaded_file = st.file_uploader("Şəkil yükləyin", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Yüklənmiş şəkil", width=250)

    with col2:
        st.write("Proqnoz üçün aşağıdakı düyməyə klikləyin.")

    if st.button("Proqnozu göstər"):
        img_array = image.img_to_array(img.resize((224, 224))) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)

        with col2:
            st.subheader("Modelin Proqnozu:")
            if prediction.shape[-1] > 1:
                st.write(f"Proqnozlaşdırılan sinif: **{np.argmax(prediction)}**")
            else:
                real_prob = prediction[0][0]
                label = "Real" if real_prob > 0.5 else "Fake"
                st.write(f"Real ehtimalı: **{real_prob:.4f}**")
                st.write(f"Fake ehtimalı: **{1 - real_prob:.4f}**")
                st.write(f"Sinif: **{label}**")
