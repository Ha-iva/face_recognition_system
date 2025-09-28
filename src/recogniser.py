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
        # self.image_dir = "Images/testing_images"
        self.output_dir = "Images/output_images"
        os.makedirs(self.output_dir, exist_ok=True)

        # Load embeddings
        embeddings_path = "face_embeddings.pickle"
        if os.path.exists(embeddings_path):
            with open(embeddings_path, "rb") as f:
                self.known_faces = pickle.load(f)
            print(f"✅ Loaded {len(self.known_faces)} embeddings.")
        else:
            self.known_faces = []
            print("⚠️ No embeddings found. Starting with an empty list.")


    def process_images(self, input_path, output_dir):
        if os.path.isfile(input_path):
            image_list = [input_path]
        else:
            image_list = [os.path.join(input_path, f) for f in os.listdir(input_path)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]

        os.makedirs(output_dir, exist_ok=True)

        count = 0
        total_images = 0

        for img_path in image_list:
            img_name = os.path.basename(img_path)
            print(f"\nProcessing: {img_name}")
            img = cv2.imread(img_path)
            if img is None:
                print(f"❌ Could not read {img_name}")
                continue

            faces = self.app.get(img)
            total_images += 1

            if faces:
                for face in faces:
                    emb1 = np.array(face['embedding']).reshape(1, -1)
                    emb1 = emb1 / np.linalg.norm(emb1)

                    x1, y1, x2, y2 = map(int, face['bbox'])

                    # Ensure coordinates are within image bounds
                    h, w = img.shape[:2]
                    x1 = max(0, min(x1, w - 1))
                    x2 = max(0, min(x2, w - 1))
                    y1 = max(0, min(y1, h - 1))
                    y2 = max(0, min(y2, h - 1))
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
                        cv2.putText(img, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                        print(f"✅ Best score: {best_score:.4f}, Found: {found}")
                    else:
                        print("❌ No similarity matches found")

                    count += 1
            else:
                print("❌ No faces detected.")

            cv2.imwrite(os.path.join(output_dir, img_name), img)

        print(f"\n✅ Total faces processed: {count}")
        print(f"✅ Total images processed: {total_images}")

    def process_video(self, video_path, output_path):
        vid_name = os.path.splitext(os.path.basename(video_path))[0]
        if output_path:
            os.makedirs(output_path, exist_ok=True)
            output_base_path = os.path.join(output_path, "output")
        else:
            output_base_path = os.path.splitext(video_path)[0] + "_output"

        output_video_file = output_base_path + ".mp4"
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("❌ Error opening video source.")
            return

        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        dim = (640, int(640 * h / w))

        out = None
        out = cv2.VideoWriter(
                output_video_file,
                cv2.VideoWriter_fourcc(*"mp4v"),
                orig_fps,
                dim
            )

        frame_count = face_processed_count = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
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
                            
                        cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 1)
                        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
                        cv2.rectangle(frame_resized, (x1, y1 - text_h - 2), (x1 + text_w + 2, y1), color, cv2.FILLED)
                        cv2.putText(frame_resized, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color2, 1)

                        print(f"✅ Best score: {best_score:.2f}, Found: {found}")
                        face_processed_count += 1

            # if show:
            #     cv2.imshow("Video Feed", frame_resized)
            #     key = cv2.waitKey(int(1000 / orig_fps)) & 0xFF
            #     if key == ord('q'):
            #         break
            # else:
            out.write(frame_resized)

        cap.release()
        if out:
            out.release()
        # if show:
        #     cv2.destroyAllWindows()

        duration = time.time() - start_time
        print("\n----- Summary -----")
        print(f"📼 Frames read: {frame_count}")
        print(f"😀 Faces Recognized: {face_processed_count}")
        print(f"⏱️ Time elapsed: {duration:.2f}s, FPS: {frame_count / duration:.2f}")

        return output_video_file  # <-- Return the actual output video path

# if __name__ == "__main__":
#     fr = FaceRecognizer()
#     # fr.process_images()  # For processing images
#     fr.process_video(video_path="Videos/testing_video.mp4")  # For processing video
#     # print("---- Recognition Completed ----")
