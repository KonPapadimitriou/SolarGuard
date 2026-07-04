import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2

# --- ΕΞΑΦΑΝΙΣΗ ΛΟΓΟΤΥΠΩΝ & MENU STREAMLIT ΓΙΑ ΑΣΦΑΛΕΙΑ ---
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
# --------------------------------------------------------

# 1. Τίτλος της σελίδας
st.title("☀️ SolarGuard: Ανίχνευση Βλαβών")

# 2. Φόρτωση του μοντέλου (έτοιμο για το Ίντερνετ)
model = YOLO("best2.pt")

# 3. Επιλογή τρόπου εισαγωγής (Κάμερα ή Ανέβασμα)
st.write("Επίλεξε πώς θέλεις να αναλύσεις το πάνελ:")
tab1, tab2 = st.tabs(["📸 Χρήση Κάμερας", "📂 Ανέβασμα Αρχείου"])

with tab1:
    camera_photo = st.camera_input("Τράβηξε μια φωτογραφία...")

with tab2:
    uploaded_photo = st.file_uploader("Ανέβασε μια φωτογραφία από το κινητό...", type=["jpg", "png", "jpeg"])

# Κρατάμε όποιο από τα δύο χρησιμοποίησε ο χρήστης
final_image = camera_photo if camera_photo is not None else uploaded_photo

# 4. Η λογική αν υπάρχει φωτογραφία (από όποια πηγή κι αν ήρθε)
if final_image is not None:
    # Εμφανίζουμε την αρχική εικόνα
    image = Image.open(final_image)
    st.image(image, caption="Προς Ανάλυση", use_container_width=True)
    
    # Φτιάχνουμε ένα κουμπί για να ξεκινήσει η ανάλυση
    if st.button("🔍 Ανάλυση Βλαβών"):
        with st.spinner("Το AI αναλύει την εικόνα..."):
            # Το μοντέλο κάνει την πρόβλεψη
            results = model.predict(source=image, conf=0.5)
            
            # Παίρνουμε την εικόνα με τα ζωγραφισμένα κουτάκια (σε μορφή BGR)
            res_image_bgr = results[0].plot()
            
            # Μετατρέπουμε τα χρώματα από BGR σε RGB
            res_image_rgb = cv2.cvtColor(res_image_bgr, cv2.COLOR_BGR2RGB)
            
            # Την εμφανίζουμε στην οθόνη με τα σωστά χρώματα
            st.image(res_image_rgb, caption="Αποτέλεσμα AI", use_container_width=True)
