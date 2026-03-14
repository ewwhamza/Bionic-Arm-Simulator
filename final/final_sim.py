import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import threading
import time
import serial
import numpy as np
import joblib

# 1 = Open/Resting, 2 = Arm Closed, 3 = One Finger Point
current_prediction = 1 
ai_initialized = False

# 1. AI PREDICTION ENGINE (Runs in the background thread)

def prediction_engine_thread():
    global current_prediction, ai_initialized
    
    print("\n--- Starting AI Initialization ---")
    
    # CONFIGURATION & SETUP
    PORT = "COM3"   
    BAUD = 115200
    WINDOW_SIZE = 200

    print("Loading emg_model4.pkl...")
    try:
        model = joblib.load("emg_model4.pkl")
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"Connecting to Arduino on {PORT}...")
    try:
        ser = serial.Serial(PORT, BAUD)
        time.sleep(2) 
    except Exception as e:
        print(f"Failed to connect to Arduino: {e}")
        return

    def extract_features(signal):
        signal = np.array(signal)
        return [np.mean(signal), np.std(signal), np.max(signal), np.min(signal), 
                np.median(signal), np.var(signal), np.sum(np.abs(np.diff(signal))), np.mean(np.abs(signal))]

    # CALIBRATION (Baseline & MVC)
    print("\n[STEP 1] Stabilizing sensor. Please wait 5 seconds...")
    end_time = time.time() + 5
    while time.time() < end_time:
        ser.readline() 

    print("\n[STEP 2] CALIBRATION 1: Keep your hand completely relaxed (REST).")
    print("Collecting baseline, wait 10 seconds")
    time.sleep(1)

    baseline_data = []
    end_time = time.time() + 10
    while time.time() < end_time:
        line = ser.readline().decode(errors='ignore').strip()
        if line.isdigit():
            baseline_data.append(float(line))

    live_baseline = np.mean(baseline_data)
    print(f"Resting baseline locked in: {live_baseline:.2f}")

    print("\n[STEP 3] CALIBRATION 2: Squeeze your fist AS HARD AS YOU CAN.")
    print("Squeeze in 3... 2... 1... SQUEEZE AND HOLD!")

    mvc_data = []
    end_time = time.time() + 3 
    while time.time() < end_time:
        line = ser.readline().decode(errors='ignore').strip()
        if line.isdigit():
            mvc_data.append(float(line))

    live_mvc = np.percentile(mvc_data, 95)
    if live_mvc <= live_baseline:
        live_mvc = live_baseline + 1.0

    print(f"MVC locked in: {live_mvc:.2f}")
    print("You can relax now!")

    # LIVE PREDICTION LOOP
    print("\n[STEP 4] LIVE PREDICTION STARTING...")
    print("Perform gestures: REST, POINT, or CLOSE. Press Ctrl+C in terminal to stop.\n")

    ai_initialized = True 

    window = []
    ser.reset_input_buffer()
    try:
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode(errors='ignore').strip()
                
                if line.isdigit():
                    raw_val = float(line)
                    norm_val = (raw_val - live_baseline) / (live_mvc - live_baseline)
                    window.append(norm_val)

                    if len(window) >= WINDOW_SIZE:
                        features = extract_features(window[:WINDOW_SIZE])
                        prediction = model.predict([features])[0]
                        
                        avg_effort = np.mean(window) * 100
                        print(f"AI Prediction: {prediction:<5} | Effort: {avg_effort:>5.1f}%")

                        if prediction == "REST":
                            current_prediction = 1
                        elif prediction == "CLOSE":
                            current_prediction = 2
                        elif prediction == "POINT":
                            current_prediction = 3

                        window = []
                        ser.reset_input_buffer() 
            else:
                time.sleep(0.001)
                    
    except KeyboardInterrupt:
        print("\nStopping AI thread...")
    finally:
        ser.close()
        print("Serial port closed.")

# 2. THE PYGAME 3D ENGINE (Runs on the main thread)
FINGER_SPEED = 8.0 

class Joint:
    def __init__(self):
        self.angle = 0.0
        self.target_angle = 0.0

    def update(self):
        self.angle += (self.target_angle - self.angle) / FINGER_SPEED

class Finger:
    def __init__(self, x_offset, is_thumb=False, side='left'):
        self.x = x_offset
        self.is_thumb = is_thumb
        self.side = side 
        self.j1 = Joint() 
        self.j2 = Joint() 
        self.j3 = Joint() 

    def update(self):
        self.j1.update(); self.j2.update(); self.j3.update()

    def draw(self):
        glPushMatrix()
        m = 1 if self.side == 'left' else -1
        
        if self.is_thumb:
            glTranslatef(-2.5 * m, -0.8, 0.4) 
            glRotatef(-20 * m, 0, 1, 0) 
            glRotatef(self.j1.angle, 0, 1, 1 * m) 
            self.draw_cuboid(0.45, 1.4, 0.45)
            glTranslatef(0, 1.4, 0)
            glRotatef(self.j2.angle, 0, 0, 1 * m) 
            self.draw_cuboid(0.4, 1.1, 0.4)
        else:
            glTranslatef(self.x, 0, 0) 
            glRotatef(self.j1.angle, 1, 0, 0)
            self.draw_cuboid(0.35, 1.4, 0.35)
            glTranslatef(0, 1.4, 0)
            glRotatef(self.j2.angle, 1, 0, 0)
            self.draw_cuboid(0.3, 1.1, 0.3)
            glTranslatef(0, 1.1, 0)
            glRotatef(self.j3.angle, 1, 0, 0)
            self.draw_cuboid(0.25, 0.8, 0.25)
            
        glPopMatrix()

    def draw_cuboid(self, w, h, d):
        glBegin(GL_QUADS)
        faces = [
            [(-w,0,d), (w,0,d), (w,h,d), (-w,h,d)], [(-w,0,-d), (w,0,-d), (w,h,-d), (-w,h,-d)],
            [(-w,h,d), (w,h,d), (w,h,-d), (-w,h,-d)], [(-w,0,d), (w,0,d), (w,0,-d), (-w,0,-d)],
            [(-w,0,d), (-w,h,d), (-w,h,-d), (-w,0,-d)], [(w,0,d), (w,h,d), (w,h,-d), (w,0,-d)]
        ]
        for face in faces:
            for v in face: glVertex3fv(v)
        glEnd()

def draw_hand_base():
    w, h, d = 2.7, 3.8, 0.7
    glColor3f(0.7, 0.5, 0.3)
    glBegin(GL_QUADS)
    for face in [[(-w,-h,d), (w,-h,d), (w,0,d), (-w,0,d)], [(-w,-h,-d), (w,-h,-d), (w,0,-d), (-w,0,-d)],
                 [(-w,-h,d), (w,-h,d), (w,-h,-d), (-w,-h,-d)], [(-w,0,d), (w,0,d), (w,0,-d), (-w,0,-d)],
                 [(-w,-h,d), (-w,0,d), (-w,0,-d), (-w,-h,-d)], [(w,-h,d), (w,0,d), (w,0,-d), (w,-h,-d)]]:
        for v in face: glVertex3fv(v)
    glEnd()

    aw, ad, wrist_h = 1.8, 0.65, 1.0
    y_p, y_w = -h, -h - wrist_h
    glColor3f(0.65, 0.45, 0.28)
    glBegin(GL_QUADS)
    glVertex3f(-w, y_p, d); glVertex3f(w, y_p, d); glVertex3f(aw, y_w, ad); glVertex3f(-aw, y_w, ad)
    glVertex3f(-w, y_p, -d); glVertex3f(w, y_p, -d); glVertex3f(aw, y_w, -ad); glVertex3f(-aw, y_w, -ad)
    glVertex3f(-w, y_p, d); glVertex3f(-aw, y_w, ad); glVertex3f(-aw, y_w, -ad); glVertex3f(-w, y_p, -d)
    glVertex3f(w, y_p, d); glVertex3f(aw, y_w, ad); glVertex3f(aw, y_w, -ad); glVertex3f(w, y_p, -d)
    glEnd()
    
    ah = 8.0
    y_bot = y_w - ah
    glColor3f(0.6, 0.4, 0.25)
    glBegin(GL_QUADS)
    for face in [[(-aw, y_bot, ad), (aw, y_bot, ad), (aw, y_w, ad), (-aw, y_w, ad)],
                 [(-aw, y_bot, -ad), (aw, y_bot, -ad), (aw, y_w, -ad), (-aw, y_w, -ad)],
                 [(-aw, y_bot, ad), (-aw, y_w, ad), (-aw, y_w, -ad), (-aw, y_bot, -ad)],
                 [(aw, y_bot, ad), (aw, y_w, ad), (aw, y_w, -ad), (aw, y_bot, -ad)],
                 [(-aw, y_bot, ad), (aw, y_bot, ad), (aw, y_bot, -ad), (-aw, y_bot, -ad)]]:
        for v in face: glVertex3fv(v)
    glEnd()

def main():
    global current_prediction, ai_initialized
    
    pygame.init()
    display = (1024, 768)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Select Hand -> Initialize AI -> Run")
    
    glEnable(GL_DEPTH_TEST)
    clock = pygame.time.Clock()
    
    state = "SELECT" 
    fingers = []
    side = "left"

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); return
            
            # STEP 1: ASK WHICH HAND
            if state == "SELECT":
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_l or event.key == pygame.K_r:
                        side = "left" if event.key == pygame.K_l else "right"
                        m = 1 if side == 'left' else -1
                        fingers = [Finger(0, is_thumb=True, side=side)] + \
                                  [Finger((i * 1.1 - 1.5) * m, side=side) for i in range(4)]
                        
                        # Transition to loading state
                        state = "LOADING"
                        pygame.display.set_caption("Initializing AI Engine... Check Terminal/Console.")
                        
                        # STEP 2: INITIALIZE AI ON A BACKGROUND THREAD 
                        ai_thread = threading.Thread(target=prediction_engine_thread, daemon=True)
                        ai_thread.start()
            
            elif state == "SIM" and event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE: 
                state = "SELECT"
                ai_initialized = False 

        if state == "LOADING" and ai_initialized:
            state = "SIM"

        # STEP 3: APPLY LIVE PREDICTIONS 
        if state == "SIM":
            if current_prediction == 1: # OPEN/REST
                for f in fingers: f.j1.target_angle = f.j2.target_angle = f.j3.target_angle = 0
            elif current_prediction == 2: # CLOSE/ARM CLOSED
                for f in fingers:
                    if f.is_thumb: f.j1.target_angle, f.j2.target_angle = -70, -50
                    else: f.j1.target_angle, f.j2.target_angle, f.j3.target_angle = 80, 90, 80
            elif current_prediction == 3: # POINT
                for i, f in enumerate(fingers):
                    if i == 1: f.j1.target_angle = f.j2.target_angle = f.j3.target_angle = 0 
                    elif f.is_thumb: f.j1.target_angle, f.j2.target_angle = -70, -50
                    else: f.j1.target_angle, f.j2.target_angle, f.j3.target_angle = 85, 90, 85

        # RENDER 
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        gluPerspective(45, (display[0]/display[1]), 0.1, 100.0)
        glTranslatef(0.0, 2.0, -25.0)
        glRotatef(-45, 1, 0, 0) 
        glRotatef(35, 0, 0, 1)

        if state == "SIM" or state == "LOADING":
            draw_hand_base()
            for f in fingers:
                f.update()
                glColor3f(0.8, 0.6, 0.4)
                f.draw()
                
            if state == "SIM":
                pygame.display.set_caption(f"Active: {side.upper()} Hand | Mode: Live EMG Tracking")

        else:
            pygame.display.set_caption("PRESS 'L' FOR LEFT HAND OR 'R' FOR RIGHT HAND")

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()