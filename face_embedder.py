import cv2
import os
import pickle
import glob
from insightface.app import FaceAnalysis

class FaceEmbedder:
    def __init__(self):
        self.INPUT_DIR = "face_recognition_system/Images/input_images"
        self.EMBEDDING_DIR = "face_recognition_system/"
        os.makedirs(self.EMBEDDING_DIR, exist_ok=True)

        self.embeddings = []
        self.labels = []
        self.all_embeddings = []

        self.app = FaceAnalysis(name="buffalo_l", allowed_modules=['detection', 'recognition'])
        self.app.prepare(ctx_id=-1,det_thresh=0.6)
        
        self.load_existing_embeddings()

    def embeddings_file(self, embedding, label, img_path):
        if embedding is None or label is None:
            print(f"Failed to process {img_path}")
        else:
            self.all_embeddings.append((embedding, label))

    def save_embeddings(self):
        pkl_path = os.path.join(self.EMBEDDING_DIR, "face_embeddings.pickle")
        with open(pkl_path, "wb") as f:
            pickle.dump(self.all_embeddings, f)
        print("Face embeddings extracted and saved successfully!", pkl_path)

    def preprocess_image(self, image, img_path):
        label = os.path.splitext(os.path.basename(img_path))[0]
        label = label.split("_")[0]
        done=False
        faces = self.app.get(image)
        if faces:
            for i in faces:
                embedding = i['embedding']
                done = True
                self.embeddings_file(embedding, label, img_path)
        else:
            print(f"No landmarks detected in image {img_path}")
        return done

    def process_single_image(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            print("Failed to load image:", img_path)
            return
        processed = self.preprocess_image(img, img_path)
        if processed:
            self.save_embeddings()
            print("Embedding for single image saved.")
        else:
            print("No face detected in the image.")

    def load_existing_embeddings(self):
        pkl_path = os.path.join(self.EMBEDDING_DIR, "face_embeddings.pickle")
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                self.all_embeddings = pickle.load(f)
            print("Loaded existing embeddings:", len(self.all_embeddings))

    def new_capture(self):
        extensions = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
        image_list = []
        for ext in extensions:
            image_list.extend(glob.glob(os.path.join(self.INPUT_DIR, ext)))
        c=0
        count = 0
        for img_path in image_list:
            img = cv2.imread(img_path)
            processed_images = self.preprocess_image(img, img_path)
            if processed_images is None:
                print("No face detected:", img_path)
                continue
            
            count += 1

        self.save_embeddings()
        print("Total number of embeddings saved:",len(self.all_embeddings))
        print("Total number of images processed:", count)
        print(c)
        
# Run the process
if __name__ == "__main__":
    face_embedder = FaceEmbedder()
    face_embedder.new_capture()
    
if __name__ == "__main__":
    face_embedder = FaceEmbedder()
    # Process a single image
    image_path = "face_recognition_system/Images/input_images/your_image.jpg"  # CHANGE THIS
    face_embedder.process_single_image(image_path)

    