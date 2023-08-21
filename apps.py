import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Function to load the model
def load_model():
    # Load the pre-trained model and weights
    model = tf.keras.models.load_model('models/trained_model.h5')

    return model

# Function to make predictions
def predict(model, image_path):
    IMAGE_SIZE = 224

    image = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = image_array / 255.0

    prediction = model.predict(image_array)
    index = np.argmax(prediction)
    confidence_score = prediction[0][index]

    return index, confidence_score

def main():
    st.title("Ornamental Plants Classification")
    st.markdown("Ornamental Plants atau Tanaman Hias adalah tanaman yang ditanam dan dipelihara untuk tujuan estetika dan keindahan.")
    st.write("Dalam website ini hanya dapat mengidentifikasi 5 jenis tanaman hias yaitu: Damask Rose, Echeveria Flower, Mirabilis Jalapa, Rain Lily, dan Zinnia Elegans.")
    st.write("Upload disini!")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load the model
        model = load_model()

        # Make predictions
        class_names = ["Damask Rose", "Echeveria Flower", "Mirabilis Jalapa", "Rain Lily", "Zinnia Elegans"]
        image_path = 'uploaded_image.jpg'
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        predicted_class, confidence_score = predict(model, image_path)

        # Display the image
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

        # Show prediction information
        st.write(f"Predicted Class: {class_names[predicted_class]}")
        st.write(f"Confidence Score: {confidence_score:.2f}")

        # Add description based on the predicted class
        if predicted_class == 0:
            st.write("Damask Rose atau Mawar Damaskus (Rosa × damascena) adalah salah satu jenis mawar yang sering ditanam sebagai tanaman hias dan bunga potong karena kecantikan dan keharumannya yang menawan. Mawar Damaskus diyakini berasal dari wilayah Timur Tengah, khususnya dari daerah pegunungan di wilayah utara Persia (sekarang Iran) atau Turki. Mawar Damaskus berwarna merah muda hingga merah tua yang indah yang telah lama dijadikan simbol keindahan dan cinta. Karena keharumannya yang halus, mawar damaskus ditanam secara luas untuk produksi minyak esensial, ataupun pembuatan teh mawar dan beberapa hidangan kuliner untuk memberikan rasa dan aroma mawar yang unik.")
        elif predicted_class == 1:
            st.write("Echeveria Flower atau Bunga Echeveria adalah bunga yang mekar pada tanamanan Echeveria. Tanaman Echeveria itu sendiri adalah genus tanaman sukulen yang terkenal karena bentuk warna daunnya yang menarik. Echeveria adalah anggota dari keluarga Crassulaceae dan berasal dari Amerika Tengah dan Selatan. Bunga Echeveria biasanya tumbuh dalam bentuk tandan dengan kelopak bunga yang berbentuk corong dengan berbagai warna seperti merah, merah muda, jingga, kuning, dan putih. Tanaman ini populer sebagai tanaman hias dan sering ditemukan dalam pot atau taman sebagai tanaman indoor atau outdoor. Bunga Echeveria muncul pada umur tertentu, tergantung pada kondisi tumbuh dan spesiesnya. Ketika bunga mekar, ia akan menambah daya tarik visual pada tanaman dan dapat menarik serangga penyerbuk seperti lebah dan kupu-kupu.")
        elif predicted_class == 2:
            st.write("Mirabilis Jalapa atau lebih dikenal sebagai Four O'Clock adalah tanaman bunga tahunan atau tanaman tahunan pendek yang berasal dari Amerika Tengah dan Selatan. Four O'Clock merujuk pada kebiasaan bunga ini membuka kelopaknya pada sekitar pukul 4 sore (juga dapat bervariasi tergantung pada kondisi cuaca dan iklim). Ciri khas utama dari Mirabilis Jalapa adalah keindahan dan variasi warna bunganya seperti merah, kuning, putih, jingga, merah muda, ungu, dan campuran warna lainnya. Mirabilis Jalapa juga dikenal karena harum wangi bunganya, terutama pada malam hari. Meskipun disebut Four O'Clock, bunga ini seringkali tidak benar-benar hanya mekar pada pukul empat sore, tetapi dapat mekar sepanjang hari. Mirabilis Jalapa dapat digunakan sebagai obat amandel karena memiliki sifat antiinflamasi dan analgesik. Selain sebagai obat amandel, Mirabilis Jalapa juga sering digunakan untuk obat demam, wasir, nyeri sendi, sariawan, dan kondisi kulit tertentu. Namun, penting untuk dicatat bahwa beberapa bagian tanaman ini mengandung senyawa toksik dan harus dihindari untuk dikonsumsi tanpa pengawasan yang tepat.")
        elif predicted_class == 3:
            st.write("Rain Lily yang juga dikenal sebagai Zephyranthes, adalah kelompok tanaman bunga kecil yang termasuk dalam keluarga Amaryllidaceae. Nama 'Rain Lily' berasal dari kebiasaan tanaman ini mekar setelah hujan dan bunga-bunganya seringkali muncul secara tiba-tiba setelah curah hujan yang baik. Tanaman Rain Lily biasanya tumbuh subur di daerah beriklim hangat dan lembap dengan variasi warna bunganya yaitu putih, merah muda, kuning, jingga, merah, dan ungu. Di Afrika Selatan, daun rain lily digunakan sebagai obat untuk diabetes melitus, manfaat lainnya terdapat pada bunga rain lily sebagai produk kecantikan yang dapat membantu meremajakan kulit dan akntioksidan. Rain Lily pastinya juga dapat memberikan keindahan dan pesona sebagai daya tarik serangga penyerbuk yang dapat membantu dalam penyerbukan dan pemuliaan tanaman.")
        elif predicted_class == 4:
            st.write("Zinnia Elegans adalah spesies tanaman bunga yang termasuk dalam genus Zinnia dalam keluarga Asteraceae. Tanaman ini sering dikenal dengan sebutan Zinnia Garden atau Zinnia Common karena popularitasnya sebagai tanaman hias di taman dan lanskap. Ciri khas utama dari Zinnia elegans adalah bunga-bunga yang indah dan beragam warna. Bunga-bunga ini memiliki kelopak berbentuk bunga matahari dan datang dalam berbagai warna cerah seperti merah, merah muda, oranye, kuning, ungu, putih, dan campuran warna lainnya. Beberapa varietas Zinnia Elegans juga memiliki bunga dua warna yang menarik. Selain menjadi pilihan favorit di kebun dan lanskap, Zinnia Elegans juga menarik serangga penyerbuk seperti lebah dan kupu-kupu, yang membantu dalam polinasi dan menjaga ekosistem tumbuhan. Zinnia Elegans juga bermanfaat dalam pengobatan seperti mengobati hepatitis, mengatasi bisul dan gatal pada kulit, dan memperlancar menstruasi.")    

if __name__ == '__main__':
    main()
