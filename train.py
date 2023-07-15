import multiprocessing
import os
import time
import pickle
import numpy as np
from aiGame import Bird, create_pipes, check_collisions, reset_game, update_game, WIDTH, HEIGHT, GROUND_HEIGHT, FPS

# Set up logging
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def train(q_table, generation_num):
    bird = Bird(None, 0, 0)
    pipes = create_pipes(None)
    score_text = None
    last_update = time.perf_counter()
    epsilon = 0.2
    game_over = False

    while not game_over:
        state = get_state(bird, pipes)
        action = get_action(state, q_table, epsilon)
        reward = get_reward(bird, pipes)
        next_state = get_state(bird, pipes, action)
        update_q_table(state, action, reward, next_state, q_table)

        if action == 1:
            bird.jump()

        bird.move(1/FPS)
        for pipe in pipes:
            pipe.move(1/FPS)
            if pipe.offscreen():
                x = max([pipe.x for pipe in pipes]) + 150
                pipe.recycle(x)
            if pipe.x + pipe.width < bird.x and not pipe.scored:
                pipe.scored = True
            if check_collisions(bird, pipes):
                logging.info(f"Collision detected! Generation: {generation_num}")
                game_over = True
                break

        if time.perf_counter() - last_update >= 1:
            last_update = time.perf_counter()
            with open('q_table.pickle', 'wb') as f:
                pickle.dump(q_table, f)
            time.sleep(np.random.uniform(0, 1))
            logging.info(f"Generation {generation_num} - State: {state}, Action: {action}, Reward: {reward}, Score: {score}")
    reset_game(bird, pipes, score_text)

def get_state(bird, pipes, action=None):
    state = {}
    state['bird_y'] = int(bird.y)
    state['bird_y_vel'] = int(bird.y_vel)
    for i, pipe in enumerate(pipes):
        state[f'pipe_{i}_x'] = int(pipe.x)
        state[f'pipe_{i}_y'] = int(pipe.y)
        state[f'pipe_{i}_scored'] = int(pipe.scored)
        state[f'pipe_{i}_distance'] = int(pipe.x - bird.x)
        state[f'pipe_{i}_height_diff'] = int(bird.y - pipe.y)
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

def get_reward(bird, pipes):
    if check_collisions(bird, pipes):
        return -1000
    for pipe in pipes:
        if pipe.x + pipe.width > bird.x and not pipe.scored:
            return 100
    return 0

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
    td_error = reward + gamma * max_next_q_value - q_values[action]
    q_values[action] += alpha * td_error
    q_table[state] = q_values

def main():
    try:
        with open('q_table.pickle', 'rb') as f:
            q_table = pickle.load(f)
        with open('generation.pickle', 'rb') as f:
            generation = pickle.load(f)
    except FileNotFoundError:
        # If the files do not exist, use default values
        q_table = {}
        generation = 1

    # Set up multiprocessing
    num_processes = multiprocessing.cpu_count()
    processes = []
    for i in range(num_processes):
        p = multiprocessing.Process(target=train, args=(q_table, generation+i))
        p.start()
        processes.append(p)

    # Wait for all processes to finish
    for p in processes:
        p.join()

    # Save the Q-table and generation number to disk
    with open('q_table.pickle', 'wb') as f:
        pickle.dump(q_table, f)
    with open('generation.pickle', 'wb') as f:
        pickle.dump(generation+num_processes, f)
        
if __name__ == '__main__':
    main()