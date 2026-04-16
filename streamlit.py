import tempfile
from streamlit_extras.grid import grid
import streamlit as st
from PIL import Image
import cv2
import base64
import os
from core.models import Models, HyperParameters


@st.cache_data
def _video_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


st.set_page_config(page_title="YOLO Inference", layout="wide")
st.title("Détection d'objets — YOLO")

# task_type = st.sidebar.selectbox('Pick a Task',HyperParameters.task_ls)


selected_model = st.sidebar.selectbox('Model', HyperParameters.model_ls)

model = Models.load_model(selected_model)
if type(model) != list:
    # ── Sidebar ──────────────────────────────────────────
    source = st.sidebar.radio("Source", ["Image", "Webcam", "Vidéo"])
    st.sidebar.markdown("---")
    conf = st.sidebar.slider('Choose a confidence score', 0.0, 1.0, 0.25, 0.01)
    HyperParameters.conf = conf
    print(HyperParameters.conf)
    print(conf)
    # ── Mode Image ───────────────────────────────────────
    if source == "Image":
        file = st.file_uploader("Charger une image", type=["jpg", "jpeg", "png", "webp"])
        if file:
            image = Image.open(file)
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original")
                st.image(image, use_container_width=True)

            with st.spinner("Inférence en cours…"):
                detections = model.predict(image, conf=HyperParameters.conf)

            with col2:
                st.subheader(f"Résultats — {len(detections[0].boxes)} objet(s)")
                st.image(cv2.cvtColor(detections[0].plot(), cv2.COLOR_BGR2RGB), use_container_width=True)

            if detections:
                st.dataframe(
                    [{"classe": d.cls[0], "confiance": d.conf[0]} for d in detections[0].boxes],
                    use_container_width=True,
                )
    # ── Mode Vidéo ───────────────────────────────────────
    elif source == "Vidéo":
        file = st.file_uploader("Charger une vidéo", type=["mp4", "avi", "mov", "mkv", "webm"])
        # ── Aperçu par défaut (aucune vidéo chargée) ──
        v1 = _video_b64(
            "videos/stock-footage-delhi-india-jul-smooth-traffic-flow-at-intersection-with-green-signal.webm")
        v2 = _video_b64("videos/detection_result.mp4")
        st.iframe(f"""
                   <div style="display:flex; gap:16px;">
                     <div style="flex:1">
                       <p style="margin:0 0 6px; font-size:13px; color:#888;">Vidéo originale</p>
                       <video src="data:video/mp4;base64,{v1}"
                              autoplay loop muted playsinline
                              style="width:100%; border-radius:8px; display:block;">
                       </video>
                     </div>
                     <div style="flex:1">
                       <p style="margin:0 0 6px; font-size:13px; color:#888;">Détection YOLO</p>
                       <video src="data:video/mp4;base64,{v2}"
                              autoplay loop muted playsinline
                              style="width:100%; border-radius:8px; display:block;">
                       </video>
                     </div>
                   </div>
               """, height=400)
        if file:
            # Prévisualisation de la vidéo originale
            st.subheader("Prévisualisation — vidéo originale")
            st.video(file)

            if st.button("Lancer la détection sur la vidéo"):
                # Sauvegarde temporaire de la vidéo d'entrée
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_in:
                    tmp_in.write(file.read())
                    input_path = tmp_in.name

                output_path = input_path.replace(".mp4", "_result.mp4")

                cap = cv2.VideoCapture(input_path)
                fps = cap.get(cv2.CAP_PROP_FPS) or 25
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

                progress = st.progress(0, text="Traitement des frames…")
                frame_idx = 0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # OpenCV BGR → PIL RGB
                    image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    detections = model(frame, conf=HyperParameters.conf)

                    writer.write(detections[0].plot())

                    frame_idx += 1
                    if total > 0:
                        progress.progress(frame_idx / total,
                                          text=f"Frame {frame_idx}/{total}")

                cap.release()
                writer.release()
                progress.empty()

                # Ré-encodage H264 pour compatibilité navigateur
                h264_path = output_path.replace("_result.mp4", "_result_h264.mp4")
                os.system(
                    f"ffmpeg -y -i {output_path} "
                    f"-vcodec libx264 -crf 23 -preset fast "
                    f"-movflags +faststart {h264_path} -loglevel error"
                )
                final_path = h264_path if os.path.exists(h264_path) else output_path

                st.subheader("Résultat — vidéo annotée")
                with open(final_path, "rb") as f:
                    video_bytes = f.read()
                st.video(video_bytes)

                st.download_button(
                    label="Télécharger la vidéo annotée",
                    data=video_bytes,
                    file_name="detection_result.mp4",
                    mime="video/mp4",
                )

                # Nettoyage fichiers temporaires
                for p in [input_path, output_path, h264_path]:
                    if os.path.exists(p):
                        os.remove(p)



    # ── Mode Webcam ──────────────────────────────────────
    elif source == "Webcam":
        frame = st.camera_input("Prendre une photo")
        if frame:
            image = Image.open(frame)
            detections = model(image, conf=HyperParameters.conf)
            st.image(cv2.cvtColor(detections[0].plot(), cv2.COLOR_BGR2RGB), use_container_width=True)
            st.write(f"{len(detections[0].boxes)} objet(s) détecté(s)")

else:
    source = st.sidebar.radio("Source", ["Image", "Webcam", "Vidéo"])
    st.sidebar.markdown("---")
    col = grid(len(model))
    if source == "Image":
        file = st.file_uploader("Charger une image", type=["jpg", "jpeg", "png", "webp"])
        if file:
            image = Image.open(file)
            for i in range(len(model)):
                detections = model[i].predict(image)
                col[i].subheader(f"Model - {HyperParameters.model_ls[i]}\n"
                                 f"Résultats— {len(detections[0].boxes)} objet(s)")
                col[i].image(cv2.cvtColor(detections[0].plot(), cv2.COLOR_BGR2RGB), use_container_width=True)
            # with col1:
            #     st.subheader("Original")
            #     st.image(image, use_container_width=True)
            #
            # with st.spinner("Inférence en cours…"):
            #
            #
            # with col2:
