import streamlit as st
import cv2
import os
import numpy as np
import pickle
import pandas as pd
from datetime import datetime
from typing import Tuple, Optional

# ---------------- Configuration ----------------
BASE_DIR = os.path.abspath('.')
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
MODEL_PATH = os.path.join(BASE_DIR, 'trainer.yml')
LABELS_PATH = os.path.join(BASE_DIR, 'labels.pickle')
ATTENDANCE_CSV = os.path.join(BASE_DIR, 'attendance.csv')

os.makedirs(DATASET_DIR, exist_ok=True)

FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)

# -------------- Helper functions ----------------
def create_recognizer():
    """Return LBPH recognizer (requires opencv-contrib-python)."""
    if not hasattr(cv2, "face"):
        raise RuntimeError("cv2.face is not available. Install opencv-contrib-python.")
    return cv2.face.LBPHFaceRecognizer_create()

def capture_frame_from_camera(timeout: int = 5) -> Optional[np.ndarray]:
    """
    Open the default webcam, try to capture a single clean frame.
    timeout: seconds to try grabbing a frame before giving up.
    Returns BGR frame or None on failure.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return None
    # warm up camera
    end_time = datetime.now().timestamp() + timeout
    frame = None
    while datetime.now().timestamp() < end_time:
        ret, f = cap.read()
        if not ret:
            continue
        # pick the last good frame
        frame = f.copy()
        # small delay
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    return frame

def detect_first_face(gray_img: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None
    return faces[0]  # x,y,w,h

def save_face_images_from_frame(frame_bgr: np.ndarray, user_id: int, user_name: str, samples: int = 5) -> Tuple[bool, str]:
    """
    From a single captured frame we attempt to extract faces and create 'samples' images by slight jittering (if needed).
    For more robust results, re-run capture several times or call multiple captures.
    """
    if frame_bgr is None:
        return False, "No frame captured from camera."
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    face_box = detect_first_face(gray)
    if face_box is None:
        return False, "No face detected in the captured frame. Ensure face is centered and lighting is adequate."
    x, y, w, h = face_box
    face_roi = gray[y:y+h, x:x+w]
    folder = f"user_{int(user_id)}_{user_name.strip().replace(' ', '_')}"
    save_dir = os.path.join(DATASET_DIR, folder)
    os.makedirs(save_dir, exist_ok=True)

    saved = 0
    # We can augment by small shifts/resize to produce multiple samples from one capture
    for i in range(samples):
        # Slight random jitter within small bounds
        jitter_x = int(np.random.uniform(-0.05, 0.05) * w)
        jitter_y = int(np.random.uniform(-0.05, 0.05) * h)
        x0 = max(0, x + jitter_x)
        y0 = max(0, y + jitter_y)
        x1 = min(gray.shape[1], x0 + w)
        y1 = min(gray.shape[0], y0 + h)
        face = gray[y0:y1, x0:x1]
        try:
            face_resized = cv2.resize(face, (200, 200))
        except Exception as e:
            continue
        fname = os.path.join(save_dir, f"img_{datetime.now().strftime('%Y%m%d%H%M%S')}_{i+1}.jpg")
        cv2.imwrite(fname, face_resized)
        saved += 1

    return True, f"Saved {saved} image(s) to {save_dir}"

def train_lbph_from_dataset() -> Tuple[bool, str]:
    recognizer = create_recognizer()
    label_ids = {}
    x_train = []
    y_labels = []
    current_id = 0

    folders = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    if not folders:
        return False, "No user folders found in dataset/. Register at least one user first."

    for folder in folders:
        folder_path = os.path.join(DATASET_DIR, folder)
        # map folder -> integer label
        if folder not in label_ids:
            label_ids[folder] = current_id
            current_id += 1
        label_id = label_ids[folder]
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            try:
                img_resized = cv2.resize(img, (200,200))
            except:
                continue
            x_train.append(img_resized)
            y_labels.append(label_id)

    if len(x_train) == 0:
        return False, "No training images available."

    recognizer.train(x_train, np.array(y_labels))
    recognizer.save(MODEL_PATH)
    with open(LABELS_PATH, 'wb') as f:
        pickle.dump(label_ids, f)
    return True, f"Training complete. Model saved to {MODEL_PATH}."

def predict_from_frame(frame_bgr: np.ndarray, conf_threshold: int = 70) -> Tuple[str, Optional[int], str]:
    """
    Predict label for captured frame. Returns (label_or_status, confidence_or_None, message)
    """
    if frame_bgr is None:
        return "No Image", None, "No image captured."

    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABELS_PATH):
        return "Model Missing", None, "Model or labels not found. Train model first."

    recognizer = create_recognizer()
    recognizer.read(MODEL_PATH)
    with open(LABELS_PATH, 'rb') as f:
        label_ids = pickle.load(f)
    inv_labels = {v: k for k, v in label_ids.items()}

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    face_box = detect_first_face(gray)
    if face_box is None:
        return "No Face", None, "No face detected in the capture."

    x, y, w, h = face_box
    roi = gray[y:y+h, x:x+w]
    try:
        roi_resized = cv2.resize(roi, (200,200))
    except Exception as e:
        return "Error", None, f"Failed to process face ROI: {e}"

    label_id, confidence = recognizer.predict(roi_resized)
    # LBPH: lower confidence => better match
    if confidence < conf_threshold:
        label_name = inv_labels.get(label_id, "Unknown")
        # mark attendance in CSV (avoid duplicate same-day entries)
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        if not os.path.exists(ATTENDANCE_CSV):
            pd.DataFrame(columns=["date", "time", "label", "name"]).to_csv(ATTENDANCE_CSV, index=False)

        df = pd.read_csv(ATTENDANCE_CSV)
        already = ((df['date'] == date_str) & (df['label'] == label_name)).any()
        if not already:
            new_row = {"date": date_str, "time": time_str, "label": label_name, "name": label_name}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(ATTENDANCE_CSV, index=False)
        return label_name, int(confidence), f"Attendance marked for {label_name} at {time_str} on {date_str}"
    else:
        return "Unknown", int(confidence), "Face not recognized. Try more training samples or adjust threshold."

def read_attendance_df() -> pd.DataFrame:
    if not os.path.exists(ATTENDANCE_CSV):
        return pd.DataFrame(columns=["date","time","label","name"])
    return pd.read_csv(ATTENDANCE_CSV)

# -------------- Streamlit UI ----------------
st.set_page_config(page_title="Face Recognition Attendance", layout="centered")

st.title("Face Recognition Attendance (Streamlit + OpenCV)")

tab = st.sidebar.radio("Select", ["Register New User", "Train Model", "Mark Attendance", "View Attendance", "Dataset Browser", "Settings"])

# ---------- Settings ----------
if tab == "Settings":
    st.header("Settings")
    st.markdown("""
    - Camera capture uses OpenCV (`cv2.VideoCapture(0)`). Make sure no other app is using the camera.
    - If `cv2.face` is missing, install `opencv-contrib-python`.
    - Run the app with: `streamlit run app.py`
    """)
    st.write("Current paths:")
    st.write(f"- dataset dir: `{DATASET_DIR}`")
    st.write(f"- model path: `{MODEL_PATH}`")
    st.write(f"- labels path: `{LABELS_PATH}`")
    st.write(f"- attendance csv: `{ATTENDANCE_CSV}`")
    st.stop()

# ---------- Register New User ----------
if tab == "Register New User":
    st.header("Register New User (Capture from Webcam)")

    col1, col2 = st.columns(2)
    with col1:
        user_id = st.number_input("User ID (integer)", min_value=1, value=1, step=1)
        user_name = st.text_input("Full name", value="")
        n_samples = st.number_input("Samples to create (per capture)", min_value=1, max_value=20, value=5, step=1)
    with col2:
        st.markdown("**Instructions:**")
        st.markdown("- Position face centered in camera.\n- Good lighting helps.\n- Click **Start Capture** to take a snapshot and save samples.")

    if st.button("Start Capture and Save"):
        with st.spinner("Capturing image from webcam..."):
            frame = capture_frame_from_camera(timeout=4)
        if frame is None:
            st.error("Failed to capture from webcam. Make sure camera is available and not used by another application.")
        else:
            # show preview
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Captured frame (preview)", use_column_width=True)
            ok, msg = save_face_images_from_frame(frame, user_id, user_name, samples=int(n_samples))
            if ok:
                st.success(msg)
                st.info(f"Folder: user_{int(user_id)}_{user_name.strip().replace(' ','_')}")
            else:
                st.error(msg)

# ---------- Train Model ----------
elif tab == "Train Model":
    st.header("Train LBPH Model")
    st.markdown("This reads images from `dataset/` and trains an LBPH recognizer.")
    if st.button("Train Now"):
        with st.spinner("Training..."):
            try:
                ok, message = train_lbph_from_dataset()
            except Exception as e:
                st.error(f"Training failed: {e}")
            else:
                if ok:
                    st.success(message)
                else:
                    st.warning(message)
    # show dataset summary
    folders = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    if folders:
        st.subheader("Detected user folders")
        for f in folders:
            count = len([n for n in os.listdir(os.path.join(DATASET_DIR, f)) if n.lower().endswith(('.jpg','.png'))])
            st.write(f"- `{f}` : {count} images")
    else:
        st.info("No users registered yet. Go to 'Register New User' tab.")

# ---------- Mark Attendance ----------
elif tab == "Mark Attendance":
    st.header("Mark Attendance (Capture & Recognize)")
    st.markdown("Click **Capture & Mark** to take a snapshot and mark attendance if matched.")
    conf_thresh = st.slider("Confidence threshold (lower = stricter, LBPH lower is better)", min_value=30, max_value=150, value=70)
    if st.button("Capture & Mark"):
        with st.spinner("Capturing & recognizing..."):
            frame = capture_frame_from_camera(timeout=4)
        if frame is None:
            st.error("Failed to capture from webcam.")
        else:
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Captured frame", use_column_width=True)
            try:
                label, conf, msg = predict_from_frame(frame, conf_threshold=int(conf_thresh))
            except Exception as e:
                st.error(f"Recognition error: {e}")
            else:
                if label in ("No Image", "Model Missing", "No Face", "Error"):
                    st.error(msg)
                elif label == "Unknown":
                    st.warning(f"{msg} (confidence={conf})")
                else:
                    st.success(f"{msg} (confidence={conf})")

# ---------- View Attendance ----------
elif tab == "View Attendance":
    st.header("Attendance Records")
    df = read_attendance_df()
    if df.empty:
        st.info("No attendance records.")
    else:
        st.dataframe(df.sort_values(["date","time"], ascending=[False,False]))

# ---------- Dataset Browser ----------
elif tab == "Dataset Browser":
    st.header("Dataset Browser")
    folders = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    if not folders:
        st.info("No user folders in dataset.")
    else:
        selected = st.selectbox("Choose user folder", options=folders)
        if selected:
            imgs = [os.path.join(DATASET_DIR, selected, f) for f in os.listdir(os.path.join(DATASET_DIR, selected)) if f.lower().endswith(('.jpg','.png'))]
            st.write(f"{len(imgs)} images")
            cols = st.columns(4)
            for i, imgp in enumerate(imgs):
                try:
                    img = cv2.imread(imgp)
                    cols[i % 4].image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True)
                except:
                    continue
