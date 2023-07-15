# flappy-bird-Qlearning-cv
This was for learning AI algorithm from scratch, the AI are not working, or maybe not yet working. 

This project is a simple implementation of Flappy Bird game with AI integration using Q-Learning and Neural Networks.

The main game used in this project is a naive implementation of flappy bird.

## Usage
1. To play the game manually, run game.py.
2. To train the AI, run the train_agent() function in main of train.py. The trained model will be saved.
3. To play the game using the AI, change the train_agent() function in train.py to play_game().

## AI Models used
train.py
- This model uses computer vision to detect the balls and the pipes for training. Q learning was used. It is not yet fully trained.
  
aiGame_ver1
- This model uses Q-Learning to learn how to play the game. It is not yet fully trained.

aiGame_ver2
- This model uses a Neural Network with Q-Learning to learn how to play the game. It is not yet fully trained.

## Acknowledgments
This project is for learning purposes only.
