import tkinter as tk
import random
import time
import numpy as np
import os
import logging
import pickle
import math

# q learning

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

WIDTH = 400
HEIGHT = 600
GROUND_HEIGHT = 50
FPS = 60
print("Tkinter version:", tk.TkVersion)

class Bird:
    def __init__(self, canvas, x, y):
        self.canvas = canvas
        self.image = canvas.create_oval(x, y, x+20, y+20, fill="yellow")
        self.y_vel = 0
        self.gravity = 2000

    def jump(self, event=None):
        self.y_vel = -100

    def move(self, dt):
        self.canvas.move(self.image, 0,  self.y_vel * dt)
        self.y_vel += self.gravity * dt

    def get_bounding_box(self):
        return self.canvas.coords(self.image)
    
    @property
    def x(self):
        return self.canvas.coords(self.image)[0]

    @property
    def y(self):
        return self.canvas.coords(self.image)[1]

class Pipe:
    gap = 200
    width = 50
    speed = 100

    def __init__(self, canvas, x, height, delay=0):
        self.canvas = canvas
        self.top = canvas.create_rectangle(x, 0, x+self.width, height, fill="green")
        self.bottom = canvas.create_rectangle(x, height+self.gap, x+self.width, HEIGHT-GROUND_HEIGHT, fill="green")
        self.delay = delay
        self.scored = False
        self.frames = 0

    def move(self, dt):
        if self.frames >= self.delay:
            self.canvas.move(self.top, self.speed * dt, 0)
            self.canvas.move(self.bottom, self.speed * dt, 0)
        else:
            self.frames += 1

    def offscreen(self):
        return self.canvas.coords(self.top)[2] < 0

    def recycle(self, x):
        self.scored = False
        height = random.randint(50, HEIGHT-GROUND_HEIGHT-self.gap-50)
        self.canvas.coords(self.top, x, 0, x+self.width, height)
        self.canvas.coords(self.bottom, x, height+self.gap, x+self.width, HEIGHT-GROUND_HEIGHT)
        self.frames = 0
        
    def get_bounding_box(self):
        return (self.canvas.coords(self.top)[1], self.canvas.coords(self.top)[0],
                self.canvas.coords(self.bottom)[3], self.canvas.coords(self.bottom)[2])
        
    @property
    def x(self):
        return self.canvas.coords(self.top)[0]

    @property
    def y(self):
        return self.canvas.coords(self.top)[3]

def create_pipe(canvas, x, height):
    return Pipe(canvas, x, height)

def create_pipes(canvas):
    pipes = [create_pipe(canvas, WIDTH + i*150, random.randint(50, HEIGHT-GROUND_HEIGHT-Pipe.gap-50)) for i in range(4)]
    return pipes

def check_collisions(bird, pipes):
    for pipe in pipes:
        if (bird.x + 20 > pipe.x and bird.x < pipe.x + pipe.width and
            (bird.y < pipe.y or bird.y + 20 > pipe.y + pipe.gap)):
            return True
    if bird.y > HEIGHT-GROUND_HEIGHT:
        print("HIT GROUND!")
        return True
    return False


def reset_game(bird, pipes, score_text):
    global score, game_over, score_flag
    game_over = False
    try:
        q_table = load_data('q_table.pickle')
        generation = load_data('generation1.pickle')
    except FileNotFoundError:
        q_table = {}
        generation = 1
    for pipe in pipes:
        pipe.recycle(WIDTH + pipe.width + pipes.index(pipe)*150)
    bird.canvas.coords(bird.image, 100, HEIGHT/2, 120, HEIGHT/2+20)
    bird.y_vel = 0
    score = 0
    for pipe in pipes:
        pipe.speed = -100
    score_text.set(f"Score: 0 Generation: {generation}")
    score_flag = True

def update_game(bird, pipes, score_text, last_update, fps_label, q_table, epsilon, generation):
    global score, game_over, score_flag

    now = time.perf_counter()
    dt = now - last_update
    pipe_speed = dt
    last_update = now

    fps = int(1 / dt)
    fps_label.config(text=f"FPS: {fps}")

    bird.move(dt)
    
    state = get_state(dt, pipe_speed, bird, pipes)
    action = get_action(state, q_table, epsilon)
    reward = get_reward(bird, pipes, score, action)
    next_state = get_state(dt, pipe_speed, bird, pipes, action)
    update_q_table(state, action, reward, next_state, q_table)
    #print(state, action, reward, next_state)
    
    for pipe in pipes:
        pipe.move(pipe_speed)

        if pipe.offscreen():
            x = max([pipe.x for pipe in pipes]) + 150
            pipe.recycle(x)

        if pipe.x + pipe.width < bird.x and not pipe.scored:
            score += 1
            score_text.set(f"Score: {score} Generation: {generation}")
            state = get_state(dt, pipe_speed, bird, pipes)
            action = get_action(state, q_table, epsilon)
            reward = get_reward(bird, pipes, score, action)
            next_state = get_state(dt, pipe_speed, bird, pipes, action)
            update_q_table(state, action, reward, next_state, q_table)
            pipe.scored = True
            score_flag = True
            
        if check_collisions(bird, pipes):
            generation += 1
            state = get_state(dt, pipe_speed, bird, pipes)
            action = get_action(state, q_table, epsilon)
            reward = get_reward(bird, pipes, score, action)
            next_state = get_state(dt, pipe_speed, bird, pipes, action)
            update_q_table(state, action, reward, next_state, q_table)
            logging.info(f"Collision detected! Generation: {generation}")
            save_data(q_table, 'q_table.pickle')
            save_data(generation, 'generation1.pickle')
            game_over = True
            reset_game(bird, pipes, score_text)


    if action == 1:
        bird.jump()
    if game_over:
        return
    
    root.after(int(1000/FPS), update_game, bird, pipes, score_text, last_update, fps_label, q_table, epsilon, generation)

def get_state(dt, pipe_speed, bird, pipes, action=None):
    state = {}
    state['dt'] = int(dt)
    state['pipe_speed'] = int(pipe_speed)
    state['bird_y'] = int(bird.y)
    state['bird_y_vel'] = int(bird.y_vel)
    state['bird_gravity'] = int(bird.gravity)
    for i, pipe in enumerate(pipes):
        state[f'pipe_{i}_x'] = int(pipe.x)
        state[f'pipe_{i}_y'] = int(pipe.y)
        state[f'pipe_{i}_scored'] = int(pipe.scored)
        state[f'pipe_{i}_distance'] = int(pipe.x - bird.x)
        top_pipe_y = pipe.y + pipe.canvas.coords(pipe.top)[3]
        bottom_pipe_y = pipe.canvas.coords(pipe.bottom)[1]
        bird_to_top_pipe_dist = bird.y - top_pipe_y
        bird_to_bottom_pipe_dist = bottom_pipe_y - bird.y
        #print(bird_to_top_pipe_dist)
        #print(bird_to_bottom_pipe_dist)
        state[f'pipe_{i}_bird_to_top'] = int(bird_to_top_pipe_dist)
        state[f'pipe_{i}_bird_to_bottom'] = int(bird_to_bottom_pipe_dist)
    if action is not None:
        state['action'] = action
    return tuple(sorted(state.items()))

def get_action(state, q_table, epsilon):
    if np.random.uniform() < epsilon:
        return np.random.randint(0, 2)
    q_values = q_table.get(state)
    if q_values is None:
        return np.random.randint(0, 2)
    return np.argmax(q_values)

def get_reward(bird, pipes, score, action):
    reward = 2
    scale = math.sqrt(score)
    
    for pipe in pipes:
        if pipe.x + pipe.width < bird.x and not pipe.scored:
            reward += 100*scale
    if check_collisions(bird, pipes):
        reward -= 100
        
    output = round(reward)
    print(output)
    return output

def update_q_table(state, action, reward, next_state, q_table, alpha=0.5, gamma=0.9):
    if state is None or next_state is None:
        return
    q_values = q_table.get(state)
    if q_values is None:
        q_values = np.zeros(2)
    next_q_values = q_table.get(next_state)
    if next_q_values is None:
        next_q_values = np.zeros(2)
    max_next_q_value = np.max(next_q_values)
    td_error = reward + gamma*max_next_q_value - q_values[action]
    q_values[action] += alpha*td_error
    q_table[state] = q_values

"""
def acquire_lock(file):
    handle = win32file._get_osfhandle(file.fileno())
    overlapped = pywintypes.OVERLAPPED()
    win32file.LockFileEx(handle, win32con.LOCKFILE_EXCLUSIVE_LOCK, 0, -0x10000, overlapped)

def release_lock(file):
    handle = win32file._get_osfhandle(file.fileno())
    overlapped = pywintypes.OVERLAPPED()
    win32file.UnlockFileEx(handle, 0, -0x10000, overlapped)
"""
def save_data(data, filename):
    with open(filename, 'wb') as f:
        #acquire_lock(f)
        pickle.dump(data, f)
        #release_lock(f)

def load_data(filename):
    with open(filename, 'rb') as f:
        #acquire_lock(f)
        data = pickle.load(f)
        #release_lock(f)
    return data

def start_game(generation):
    global root, score, game_over

    root = tk.Tk()
    root.title("Flappy Bird")
    canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg="skyblue", bd=0, highlightthickness=0)
    canvas.pack()
    ground = canvas.create_rectangle(0, HEIGHT-GROUND_HEIGHT, WIDTH, HEIGHT, fill="brown")
    bird = Bird(canvas, 500, HEIGHT/2)
    root.bind("<space>", bird.jump)
    pipes = create_pipes(canvas)

    score = 0
    score_text = tk.StringVar()
    score_text.set("Score: 0")
    score_label = tk.Label(root, textvariable=score_text, font=("Helvetica", 20))
    score_label.pack()

    fps_label = tk.Label(root, text="FPS: 0", bg='black', fg='white', font=('Arial', 12))
    fps_label.pack()
    fps_label.place(x=10, y=10)

    q_table = {}
    epsilon = 0.1

    reset_game(bird, pipes, score_text)
    last_update = time.perf_counter()
    root.after(int(1000/FPS), update_game, bird, pipes, score_text, last_update, fps_label, q_table, epsilon, generation)
    root.mainloop()

def run():
    try:
        q_table = load_data('q_table.pickle')
    except FileNotFoundError:
        q_table = {}
    try:
        generation = load_data('generation1.pickle')
    except:
        generation = 1
        
    game_over = False
    start_game(generation)

if __name__ == "__main__":
    run()