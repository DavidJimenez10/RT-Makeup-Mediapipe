import cv2
import mediapipe as mp



def main():
    stream = cv2.VideoCapture(0)
    
    while True:
        ret, frame = stream.read()
        
        cv2.imshow('video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    stream.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()