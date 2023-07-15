import tkinter as tk
import random 
import time

# base game

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
        self.gravity = 1000

    def jump(self, event=None):
        self.y_vel = -300

    def move(self, dt):
        self.canvas.move(self.image, 0, self.y_vel * dt)
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
    speed = -100

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
    return False


def reset_game(bird, pipes, score_text):
    global score, game_over
    game_over = False
    time.sleep(0.5)
    for pipe in pipes:
        pipe.recycle(WIDTH + pipe.width + pipes.index(pipe)*150)
    bird.canvas.coords(bird.image, 100, HEIGHT/2, 120, HEIGHT/2+20)
    bird.y_vel = 0
    score = 0
    for pipe in pipes:
        pipe.speed = -100
    score_text.set("Score: 0")

def update_game(bird, pipes, score_text, last_update, fps_label):
    global score, game_over,score_flag

    if game_over:
        return
    
    now = time.perf_counter()
    dt = (now - last_update)
    last_update = now

    fps = int(1 / dt)
    fps_label.config(text=f"FPS: {fps}")


    bird.move(dt)
    for pipe in pipes:
        pipe.move(dt)


        if pipe.offscreen():          
            min_offset = len(pipes) * 0
            max_offset = len(pipes) * 100
            offset = random.randint(min_offset, max_offset)
            next_pipe_x = WIDTH + offset
            min_x_gap = 150
            
            if pipes:
                last_pipe = pipes[-1]
                if next_pipe_x < last_pipe.x + last_pipe.width + min_x_gap:
                    next_pipe_x = last_pipe.x + last_pipe.width + min_x_gap

                while next_pipe_x < last_pipe.x + last_pipe.width:
                    next_pipe_x += min_x_gap

            max_x_gap = WIDTH
            if next_pipe_x > max_x_gap:
                next_pipe_x = max_x_gap

            pipe.recycle(next_pipe_x + 150)

    if check_collisions(bird, pipes):
        game_over = True
        reset_game(bird, pipes, score_text)
        

    if bird.y >= HEIGHT - GROUND_HEIGHT:
        game_over = True
        reset_game(bird, pipes, score_text)

    for pipe in pipes:
        if pipe.x + pipe.width < bird.x and not pipe.scored:
            score += 1
            pipe.scored = True
            #pipe.recycle(WIDTH + pipe.width + len(pipes)*random.randint(10,30))

    score_text.set(f"Score: {score}")

    root.after(max(0, int(1000/FPS - dt*1000)), update_game, bird, pipes, score_text, now, fps_label)


def start_game():
    global root, score, game_over

    root = tk.Tk()
    root.title("Flappy Bird")
    root.geometry("400x630+0+0")
    
    canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg="skyblue", bd=0, highlightthickness=0)
    canvas.pack()
    ground = canvas.create_rectangle(0, HEIGHT-GROUND_HEIGHT, WIDTH, HEIGHT, fill="brown")
    bird = Bird(canvas, 100, HEIGHT/2)
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

    reset_game(bird, pipes, score_text)

    last_update = time.perf_counter()
    update_game(bird, pipes, score_text, last_update, fps_label)
    
    root.mainloop()

def run():
    game_over = False
    start_game()

if __name__ == "__main__":
    run()