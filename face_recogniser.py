import cv2
import numpy as np
import os
import pickle
import time
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis


class FaceRecognizer:
    def __init__(self):
        # Load InsightFace model
        self.app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"])
        self.app.prepare(ctx_id=-1, det_thresh=0.6)

        # Directories
        self.image_dir = "face_recognition_system/Images/testing_images"
        self.output_dir = "face_recognition_system/Images/output_images"
        os.makedirs(self.output_dir, exist_ok=True)

        # Load embeddings
        embeddings_path = "face_recognition_system/face_embeddings.pickle"
        if not os.path.exists(embeddings_path):
            raise FileNotFoundError(f"Embeddings not found at {embeddings_path}")
        with open(embeddings_path, "rb") as f:
            self.known_faces = pickle.load(f)

    def process_images(self):
        count = 0
        total_images = 0
        images_list = os.listdir(self.image_dir)

        for img_name in images_list:
            print(f"\nProcessing: {img_name}")
            img_path = os.path.join(self.image_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Could not read {img_name}")
                continue

            faces = self.app.get(img)
            total_images += 1

            if faces:
                for face in faces:
                    emb1 = np.array(face['embedding']).reshape(1, -1)
                    emb1 = emb1 / np.linalg.norm(emb1)

                    x1, y1, x2, y2 = map(int, face['bbox'])
                    sim = {}

                    for emb2, name in self.known_faces:
                        emb2 = emb2 / np.linalg.norm(emb2)
                        similarity = cosine_similarity(emb1, emb2.reshape(1, -1))[0][0]
                        sim[similarity] = name

                    if sim:
                        best_score = max(sim)
                        found = sim[best_score]

                        if best_score > 0.4:
                            label = f"{found} ({best_score:.2f})"
                            color = (0, 255, 0)
                        else:
                            label = "Unknown"
                            color = (0, 0, 255)

                        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                        print(f"Best score: {best_score:.4f}, Found: {found}")
                    else:
                        print("No similarity matches found")

                    count += 1
            else:
                print("No faces detected.")

            cv2.imwrite(os.path.join(self.output_dir, img_name), img)

        print(f"\n✅ Total faces processed: {count}")
        print(f"✅ Total images processed: {total_images}")

    def process_video(self, video_path, mode='none', skip_frames=5, target_fps=5):
        vid_name = os.path.splitext(os.path.basename(video_path))[0]
        FACE_DIR = os.path.join("face_recognition_system", "Videos", f"{vid_name}_{mode}")
        os.makedirs(FACE_DIR, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error opening video source.")
            return

        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        dim = (640, int(640 * h / w))

        output_fps = orig_fps
        out = cv2.VideoWriter(f"{FACE_DIR}.avi", cv2.VideoWriter_fourcc(*"mp4v"), output_fps, dim)

        repeat_factor = 1
        if mode == 'fps' and 0 < target_fps < orig_fps:
            repeat_factor = int(round(orig_fps / target_fps))

        frame_count = face_processed_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if mode == 'skip' and (frame_count % skip_frames != 0):
                continue

            frame_resized = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
            faces = self.app.get(frame_resized)

            if faces:
                for face in faces:
                    emb1 = np.array(face['embedding']).reshape(1, -1)
                    emb1 = emb1 / np.linalg.norm(emb1)

                    sim = {}
                    for emb2, name in self.known_faces:
                        emb2 = emb2 / np.linalg.norm(emb2)
                        similarity = cosine_similarity(emb1, emb2.reshape(1, -1))[0][0]
                        sim[similarity] = name

                    if sim:
                        best_score = max(sim)
                        found = sim[best_score]
                        label = f"{found} ({best_score:.2f})" if best_score >= 0.4 else "Unknown"
                        color = (0, 255, 0) if best_score >= 0.4 else (0, 0, 255)
                        color2 = (0, 0, 0) if best_score >= 0.4 else (255, 255, 255)

                        x1, y1, x2, y2 = map(int, face['bbox'])

                        if best_score >= 0.4:
                            label_clean = found.replace(" ", "_")
                            save_dir = os.path.join(FACE_DIR, label_clean)
                            os.makedirs(save_dir, exist_ok=True)

                            face_img = frame_resized[y1:y2, x1:x2]
                            new_size = (int(face_img.shape[1] * 1.5), int(face_img.shape[0] * 1.5))
                            face_resized = cv2.resize(face_img, new_size, interpolation=cv2.INTER_LINEAR)
                            filename = f"{time.time():.6f}.jpg"
                            cv2.imwrite(os.path.join(save_dir, filename), face_resized)

                        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 1)
                        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
                        cv2.rectangle(frame_resized, (x1, y1 - text_h - 2), (x1 + text_w + 2, y1), color, cv2.FILLED)
                        cv2.putText(frame_resized, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color2, 1)

                        print(f"Best score: {best_score:.2f}, Found: {found}")
                        face_processed_count += 1

            for _ in range(repeat_factor):
                out.write(frame_resized)

            cv2.imshow("Video Feed", frame_resized)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        duration = time.time() - start_time
        print("\n----- Summary -----")
        print(f"Frames read: {frame_count}")
        print(f"Faces Recognized: {face_processed_count}")
        print(f"Time elapsed: {duration:.2f}s, FPS: {frame_count / duration:.2f}")


if __name__ == "__main__":
    fr = FaceRecognizer()
    fr.process_images()  # For processing images
    # fr.process_video(video_path="videos/Evening_1.mp4", mode='fps')  # For processing video
    # print("---- Recognition Completed ----")
