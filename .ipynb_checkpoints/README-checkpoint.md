\# AlphaZero Gomoku

An implementation of Alphazero for \[Gomoku](https://en.wikipedia.org/wiki/Gomoku). Gomoku is a game very similar to Go but much simpler. Therefore we can train a good model in less time. This repo refers Monte Carlo Search Tree logics from \[junxiaosong/AlphaZero\_Gomoku](https://github.com/junxiaosong/AlphaZero\_Gomoku). Other components are redesigned including the model, the game and the way to play with the model. A lot of improvements are specifically according to the Gomoku rule to reduce the mcts search space.



\## Dependencies

python 3.9.21

tensorflow 2.10.0

numpy 1.26.4

cudnn 64_8



\## How to play

* Start the game server:

```

python backend.py

```

* Then open the index.html to play against the model. You can modify the backend.py to play different model.
