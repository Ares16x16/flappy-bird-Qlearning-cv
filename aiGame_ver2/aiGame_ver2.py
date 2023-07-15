import tkinter as tk
import random
import time
import numpy as np
import os
import logging
import pickle
import math
import tensorflow as tf
from tensorflow.python.keras import backend as K
from keras.callbacks import CSVLogger
import sys

# neural network with q learning
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config) 
K.set_session(sess)
#csv_logger = CSVLogger('log.csv', append=True, separator=';')
logging.info(tf.config.list_physical_devices('GPU'))

state_dim = 45
action_dim = 2
NUM_FEATURES = state_dim


WIDTH = 400
HEIGHT = 600
GROUND_HEIGHT = 50
FPS = 60

num_episodes = 1000
epsilon = 1.0
gamma = 0.99
alpha = 0.01
    
#print("Tkinter version:", tk.TkVersion)
print("Running")

class Bird:
    def __init__(self, canvas, x, y):
        self.canvas = canvas
        self.image = canvas.create_oval(x, y, x+20, y+20, fill="yellow")
        self.y_vel = 0
        self.gravity = 1000

    def jump(self, event=None):
        self.y_vel = -200

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
    gap = 250
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


def reset_game(bird, pipes, score_text, generation):
    global score, game_over, score_flag
    game_over = False
    for pipe in pipes:
        pipe.recycle(WIDTH + pipe.width + pipes.index(pipe)*150)
    bird.canvas.coords(bird.image, 100, HEIGHT/2, 120, HEIGHT/2+20)
    bird.y_vel = 0
    score = 0
    for pipe in pipes:
        pipe.speed = -100
    score_text.set(f"Score: 0 Generation: {generation}")
    score_flag = True

def update_game(bird, pipes, score_text, last_update, fps_label, epsilon, generation,gamma):
    global score, game_over, score_flag
    
    total_reward = 0
    
    now = time.perf_counter()
    dt = (now - last_update)
    pipe_speed = dt
    last_update = now

    # FPS
    fps = int(1 / dt)
    fps_label.config(text=f"FPS: {fps}")

    bird.move(dt)
         
    state_dict = get_state(dt, pipe_speed, bird, pipes, action=None)
    state_array = np.array(list(state_dict.values())).reshape((1, NUM_FEATURES))
    done = False
    action = get_action(dt, pipe_speed, bird, pipes, epsilon, model) 
    
    for i, pipe in enumerate(pipes):
        pipe.move(pipe_speed)
        if pipe.offscreen():
            x = max([pipe.x for pipe in pipes]) + 150
            pipe.recycle(x)
            
        # Reward setting
        if i == 0 and pipe.y + (pipe.gap)/2 - 20 < bird.y < pipe.y + (pipe.gap)/2 + 20:
            total_reward += 50
        elif i == 0 and bird.y < pipe.y:
            print("Low")
            total_reward -= 50
        elif i == 0 and bird.y > pipe.y+pipe.gap:
            total_reward -= 50
            print("High")
        elif bird.y > 550:
            total_reward -= 50
  
        #total_reward += (bird.x - pipe.x + pipe.width)*0.01
        
        if pipe.x + pipe.width < bird.x and not pipe.scored:
            score += 1
            if score == 1:
                total_reward += 1000
            score_text.set(f"Score: {score} Generation: {generation}")
            pipe.scored = True
            score_flag = True
            
        if check_collisions(bird, pipes):
            generation += 1
            game_over = True
            total_reward -= 100
            next_state_dict = get_state(dt, pipe_speed, bird, pipes, action=None)
            next_state_array = np.array(list(next_state_dict.values())).reshape((1, NUM_FEATURES))
            update_q_table(state_array, action, total_reward, next_state_array, done, gamma, alpha)
            if generation%20 == 0:
                save_data(generation,'generation.pickle')
                model.save(f'step_{generation}.keras')
                try:
                    os.remove(f'step_{generation-100}.keras')
                    print(f"Removed step_{generation-100}.keras")
                except:
                    pass
                if generation%200 == 0:
                    os.execl(sys.executable, sys.executable, *sys.argv)
            reset_game(bird, pipes, score_text, generation)

 
    total_reward += score * 100
    total_reward *= gamma
    total_reward += 10 
    
    next_state_dict = get_state(dt, pipe_speed, bird, pipes, action=None)
    next_state_array = np.array(list(next_state_dict.values())).reshape((1, NUM_FEATURES))
    update_q_table(state_array, action, total_reward, next_state_array, done, gamma, alpha)
    state_array = next_state_array   
    
    if action == 1:
        bird.jump()
    if game_over:
        return
    
    print(f"Episode {generation}: Total Reward = {total_reward} Score: {score}")
    epsilon = max(0.1, epsilon * 0.99)
    
    root.after(int(1000/FPS), update_game, bird, pipes, score_text, last_update, fps_label, epsilon, generation,gamma)

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
        state[f'pipe_{i}_bird_to_top'] = int(bird_to_top_pipe_dist)
        state[f'pipe_{i}_bird_to_bottom'] = int(bird_to_bottom_pipe_dist)
        state[f'pipe_{i}_top_width'] = int(pipe.canvas.coords(pipe.top)[2] - pipe.canvas.coords(pipe.top)[0])
        state[f'pipe_{i}_bottom_width'] = int(pipe.canvas.coords(pipe.bottom)[2] - pipe.canvas.coords(pipe.bottom)[0])
        state[f'pipe_{i}_top_height'] = int(pipe.canvas.coords(pipe.top)[3] - pipe.canvas.coords(pipe.top)[1])
        state[f'pipe_{i}_bottom_height'] = int(pipe.canvas.coords(pipe.bottom)[3] - pipe.canvas.coords(pipe.bottom)[1])
    if action is not None:
        state['action'] = action
    return dict(sorted(state.items()))

def get_action(dt, pipe_speed, bird, pipes, epsilon, model):
    state_dict = get_state(dt, pipe_speed, bird, pipes)
    state_array = np.array(list(state_dict.values())).reshape((1, NUM_FEATURES))
    if np.random.uniform() < epsilon:
        return np.random.randint(0, 2)
    q_values = model.predict(state_array)
    #print(np.argmax(q_values))
    return np.argmax(q_values)

def get_reward(bird, pipes, score, action):
    # base reward
    reward = 10
    scale = math.sqrt(score) if score != 0 else 1

    if action != 1:
        reward += 50*scale
    else:
        reward += -50*scale

    output = round(reward)
    return 0

def update_q_table(state, action, reward, next_state, done, gamma, alpha):
    target = reward
    if not done:
        next_q_values = model.predict(next_state)[0]
        next_action = np.argmax(next_q_values)
        target += gamma * next_q_values[next_action]
    q_values = model.predict(state)[0]
    q_values[action] = (1 - alpha) * q_values[action] + alpha * target
    model.fit(np.array([state][0]), np.array([q_values]), batch_size=1, epochs=1, verbose=0)

def save_data(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def start_game(generation):
    global root, score, game_over

    root = tk.Tk()
    root.title("Flappy Bird")

    canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg="skyblue", bd=0, highlightthickness=0)
    canvas.pack()
    ground = canvas.create_rectangle(0, HEIGHT-GROUND_HEIGHT, WIDTH, HEIGHT, fill="brown")
    bird = Bird(canvas, 500, HEIGHT/2)

    #root.bind("<space>", bird.jump)

    pipes = create_pipes(canvas)
    
    score = 0
    score_text = tk.StringVar()
    score_text.set("Score: 0")
    score_label = tk.Label(root, textvariable=score_text, font=("Helvetica", 20))
    score_label.pack()

    fps_label = tk.Label(root, text="FPS: 0", bg='black', fg='white', font=('Arial', 12))
    fps_label.pack()
    fps_label.place(x=10, y=10)

    epsilon = 0.1

    reset_game(bird, pipes, score_text, generation)
    last_update = time.perf_counter()
    root.after(int(1000/FPS), update_game, bird, pipes, score_text, last_update, fps_label, epsilon, generation,gamma)
    root.mainloop()

def run(generation):       
    game_over = False
    start_game(generation)

if __name__ == "__main__":
    
    try:
        generation = load_data('generation.pickle')
        model = tf.keras.models.load_model(f'step_{generation}.keras')
    except:
        generation = 0
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(32, input_dim=state_dim, activation='relu'))
        model.add(tf.keras.layers.Dense(16, activation='relu'))
        model.add(tf.keras.layers.Dense(action_dim, activation='relu'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam())

    logging.info(f"Loaded step_{generation}.keras")
    model.summary()
    run(generation)