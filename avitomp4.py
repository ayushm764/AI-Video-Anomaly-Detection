import cv2

# your input file (IMPORTANT: use raw string r"")
input_file = r"C:\Users\ASUS\OneDrive\Desktop\14.avi"
output_file = r"C:\Users\ASUS\OneDrive\Desktop\14.mp4"

cap = cv2.VideoCapture(input_file)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)

cap.release()
out.release()

print("✅ Converted to 14.mp4")