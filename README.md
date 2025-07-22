#AlphaZero Go-Moku Model

This project implements a Go-Moku (also known as Five-in-a-Row or Gobang) AI model based on the principles of the AlphaZero algorithm. Go-Moku is a board game similar to Go, but significantly simpler. This simplification allows us to train a reasonably performant model even with limited computational resources and processing power.

The core Monte Carlo Tree Search (MCTS) implementation in this project is from the [junxiaosong/AlphaZero_Gomoku](https://github.com/junxiaosong/AlphaZero_Gomoku) repository. Building upon that foundation, I've introduced several key optimizations and enhancements:

##Optimizations & Features

Refined Model Architecture:
The neural network model has been redesigned to use one Conv2D layer followed by five residual blocks. This architecture aims to improve the model's feature extraction and learning capabilities.

Go-Moku Specific Rule Handling for MCTS Efficiency:
We've implemented special handling for classical Go-Moku rules, specifically concerning patterns like "three-in-a-row" (活三) and "four-in-a-row" (冲四). By incorporating these rules into the Monte Carlo Tree Search, we've significantly optimized the MCTS search space, allowing it to focus more effectively on strategically relevant moves. This greatly improves the efficiency and quality of the search, especially in critical game phases.

Accelerated Training with Draw Condition for Long Games:
Based on known Go-Moku characteristics and training experience, we treat games exceeding a certain number of moves as a draw. This heuristic significantly accelerates the training process. In Go-Moku, games that go on for an extremely long time often lead to repetitive or unproductive states, and marking them as draws helps the model converge faster without needing to simulate every single move to its theoretical end.

Value Discounting for Shorter Wins:
A discount factor has been incorporated into the reward system. This means that games won with fewer moves are assigned a higher value, encouraging the model to find more efficient and quicker winning strategies.

Multi-threaded Training Support:
The training process now supports multi-threading to optimize resource utilization and prevent Out Of Memory (OOM) errors, allowing for more stable and extended training sessions.

Simplified Game Play for Evaluation:
For simplified self-play or evaluation, the model now randomly selects a move from the highest probability positions instead of strictly choosing the single highest probability move. This adds a slight element of exploration or variety during automated play.

##Dependencies
Python: 3.9.21

TensorFlow: 2.10.0

NumPy: 1.26.4

cuDNN: 64_8 (for GPU acceleration with TensorFlow)

All dependencies are in requirements.txt

##How to Run

Train the Model:
To start the self-play and training process:

```
python train.py
```

Play Against the AI (Web Interface):
First, start the game server:
```
python backend.py
```

Then, open the index.html file in your web browser to play against the AI through the web interface.

##Current Model Status

The current model has been trained for 160 hours on an 11x11 Go-Moku board.

##Future Plans
Will continuing optimize the program and train model for larger board.

##License
This project is open-sourced under the MIT License.

##Contact
If you have any questions or feedback, feel free to open an issue on GitHub.