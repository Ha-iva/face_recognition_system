from embedder import FaceEmbedder
from recogniser import FaceRecognizer
import os

def validate_path(path, must_exist=True, is_file=None):
    if must_exist and not os.path.exists(path):
        print(f"❌ Path does not exist: {path}")
        return False
    if is_file is not None:
        if is_file and not os.path.isfile(path):
            print(f"❌ Expected a file, got a directory: {path}")
            return False
        if not is_file and not os.path.isdir(path):
            print(f"❌ Expected a directory, got a file: {path}")
            return False
    return True

def main():
    embedder = FaceEmbedder()
    recogniser = FaceRecognizer()

    while True:
        print("\n----------------------------------------------------------")
        print("-------- Welcome to the Face Recognition System ----------")
        print("--------- Select an option from the menu below -----------")
        print("1. Save (add) new face(s)")
        print("2. Recognize face(s)")
        print("0. Exit")
        
        choice = input("Choose (0, 1, or 2): ").strip()

        if choice == "0":
            print("👋 Exiting the system. Goodbye!")
            break

        elif choice == "1":
            while True:
                print("\n🧠 Add Faces")
                print("1. Add multiple faces from a folder")
                print("2. Add a single image (filename = person's name)")
                print("0. Back to main menu")

                c2 = input("Choose (0, 1 or 2): ").strip()

                if c2 == "0":
                    break
                elif c2 == "1":
                    input_path = input("Enter folder path: ").strip()
                    if validate_path(input_path, must_exist=True, is_file=False):
                        embedder.new_capture(input_path)
                elif c2 == "2":
                    input_path = input("Enter image path: ").strip()
                    if validate_path(input_path, must_exist=True, is_file=True):
                        embedder.process_single_image(input_path)
                else:
                    print("❌ Invalid option for saving faces.")

        elif choice == "2":
            while True:
                print("\n🧠 Recognize Faces")
                print("1. From image(s)")
                print("2. From video")
                print("0. Back to main menu")

                c3 = input("Choose (0, 1 or 2): ").strip()

                if c3 == "0":
                    break
                elif c3 == "1":
                    input_path = input("Enter image path or folder: ").strip()
                    output_path = input("Enter output folder to save results: ").strip()
                    os.makedirs(output_path, exist_ok=True)
                    if validate_path(input_path, must_exist=True):
                        recogniser.process_images(input_path, output_path)
                elif c3 == "2":
                    input_path = input("Enter video path: ").strip()
                    if not validate_path(input_path, must_exist=True, is_file=True):
                        continue
                    output_path = input("Enter output path: ").strip()
                    if not output_path:
                        output_path = os.path.dirname(input_path)
                    os.makedirs(output_path, exist_ok=True)
                    recogniser.process_video(input_path, output_path)
                else:
                    print("❌ Invalid option for recognition.")
        else:
            print("❌ Invalid main option. Please choose 0, 1, or 2.")

if __name__ == "__main__":
    main()
