import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

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
        # Mirroring factor: 1 for Left, -1 for Right
        m = 1 if self.side == 'left' else -1
        
        if self.is_thumb:
            glTranslatef(-2.5 * m, -0.8, 0.4) 

            glRotatef(-20 * m, 0, 1, 0) 
            
            # JOINT 1
            glRotatef(self.j1.angle, 0, 1, 1 * m) 
            self.draw_cuboid(0.45, 1.4, 0.45)
            
            # JOINT 2
            glTranslatef(0, 1.4, 0)
            glRotatef(self.j2.angle, 0, 0, 1 * m) 
            self.draw_cuboid(0.4, 1.1, 0.4)
        else:
            # Standard Fingers
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

    # Wrist Taper
    aw, ad, wrist_h = 1.8, 0.65, 1.0
    y_p, y_w = -h, -h - wrist_h
    glColor3f(0.65, 0.45, 0.28)
    glBegin(GL_QUADS)
    glVertex3f(-w, y_p, d); glVertex3f(w, y_p, d); glVertex3f(aw, y_w, ad); glVertex3f(-aw, y_w, ad)
    glVertex3f(-w, y_p, -d); glVertex3f(w, y_p, -d); glVertex3f(aw, y_w, -ad); glVertex3f(-aw, y_w, -ad)
    glVertex3f(-w, y_p, d); glVertex3f(-aw, y_w, ad); glVertex3f(-aw, y_w, -ad); glVertex3f(-w, y_p, -d)
    glVertex3f(w, y_p, d); glVertex3f(aw, y_w, ad); glVertex3f(aw, y_w, -ad); glVertex3f(w, y_p, -d)
    glEnd()
    
    # Arm
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
    pygame.init()
    display = (1024, 768)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("L: Left Hand | R: Right Hand")
    
    glEnable(GL_DEPTH_TEST)
    clock = pygame.time.Clock()
    
    state = "SELECT" 
    fingers = []
    side = "left"

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); return
            
            if state == "SELECT":
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_l:
                        side = "left"
                        # Index to Pinky positions for Left Hand
                        fingers = [Finger(0, is_thumb=True, side=side)] + \
                                  [Finger(i * 1.1 - 1.5, side=side) for i in range(4)]
                        state = "SIM"
                    elif event.key == pygame.K_r:
                        side = "right"
                        # Index to Pinky positions for Right Hand
                        fingers = [Finger(0, is_thumb=True, side=side)] + \
                                  [Finger(-(i * 1.1 - 1.5), side=side) for i in range(4)]
                        state = "SIM"
            
            elif state == "SIM":
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE: state = "SELECT"
                    if event.key == pygame.K_1: # OPEN
                        for f in fingers: f.j1.target_angle = f.j2.target_angle = f.j3.target_angle = 0
                    elif event.key == pygame.K_2: # CLOSE
                        for f in fingers:
                            if f.is_thumb: f.j1.target_angle, f.j2.target_angle = -70, -50
                            else: f.j1.target_angle, f.j2.target_angle, f.j3.target_angle = 80, 90, 80
                    elif event.key == pygame.K_3: # POINT
                        for i, f in enumerate(fingers):
                            if i == 1: f.j1.target_angle = f.j2.target_angle = f.j3.target_angle = 0 # Index finger
                            elif f.is_thumb: f.j1.target_angle, f.j2.target_angle = -70, -50
                            else: f.j1.target_angle, f.j2.target_angle, f.j3.target_angle = 85, 90, 85

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        gluPerspective(45, (display[0]/display[1]), 0.1, 100.0)
        glTranslatef(0.0, 2.0, -25.0)
        glRotatef(-45, 1, 0, 0) 
        glRotatef(35, 0, 0, 1)

        if state == "SIM":
            draw_hand_base()
            for f in fingers:
                f.update()
                glColor3f(0.8, 0.6, 0.4)
                f.draw()
            pygame.display.set_caption(f"Active: {side.upper()} Hand (1,2,3 for Poses | ESC to Reset)")
        else:
            pygame.display.set_caption("PRESS 'L' FOR LEFT HAND OR 'R' FOR RIGHT HAND")

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()