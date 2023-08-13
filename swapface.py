import cv2

def detect_face(image_path):
    # Load the face detection classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Read the input image
    image = cv2.imread(image_path)

    # Check if the image was successfully loaded
    if image is None:
        print(f"Error: Could not read the image from '{image_path}'.")
        return None

    # Convert the image to grayscale (face detection requires grayscale)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

    # Assuming there's only one face in the image, return its coordinates
    if len(faces) == 1:
        return faces[0]
    else:
        return None

def main():
    # Path to the two input images
    image_path_1 = 'path_to_image1.jpg'
    image_path_2 = 'path_to_image2.jpg'
    

    # Detect the face in the second image
    face_coords_2 = detect_face(image_path_2)
    if face_coords_2 is None:
        print("No face found in the second image.")
        return

    # Load the first image
    image_1 = cv2.imread(image_path_1)

    # Check if the first image was successfully loaded
    if image_1 is None:
        print(f"Error: Could not read the image from '{image_path_1}'.")
        return

    # Resize the first image to match the size of the face in the second image
    face_width, face_height = face_coords_2[2], face_coords_2[3]
    image_1_resized = cv2.resize(image_1, (face_width, face_height))

    # Get the region of interest (ROI) from the second image where the face was detected
    image_2 = cv2.imread(image_path_2)
    roi = image_2[face_coords_2[1]:face_coords_2[1]+face_height, face_coords_2[0]:face_coords_2[0]+face_width]

    # Flip the ROI horizontally (reflection transformation)
    reflected_roi = cv2.flip(roi, 1)

    # Blend the reflected ROI onto the first image
    alpha = 0.7  # Adjust the blending strength here (0.0 to 1.0)
    blended_image = cv2.addWeighted(image_1_resized, alpha, reflected_roi, 1 - alpha, 0)

    # Replace the ROI in the second image with the blended image
    image_2[face_coords_2[1]:face_coords_2[1]+face_height, face_coords_2[0]:face_coords_2[0]+face_width] = blended_image

    # Display the result
    cv2.imshow('Blended Image', image_2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 
