import numpy as np
import cv2

clip = np.load("data/processed/UCSDped2/Train/Train001/clip_0010.npy")

for frame in clip:
    frame = (frame * 255).astype("uint8")
    cv2.imshow("Clip Preview", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
