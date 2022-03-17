## AlphaZero-Gobang-3players
This is an implementation of the AlphaZero algorithm for playing the simple board game Gobang for 3 players. 

References:   
~ Import 2-player source code from: https://github.com/junxiaosong/AlphaZero_Gomoku  
~ Combine GUI from: https://github.com/zouyih/AlphaZero_Gomoku-tensorflow  
~ AlphaZero: Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm  
~ AlphaGo Zero: Mastering the game of Go without human knowledge  

using pharse to control train.py.
```python
python train.py --weights path_to_model.model --save_dir models_name --width 8 --n_playout 1200 --init_playout 3000 --resume --init_batch 2700 --max_batch 5000 --res_num 2 --use_gpu
```

Using human_play.py to play with agents. Play with AlphaZero agent, with board size 8, number of playout 1200, 2 residual blocks (should match model file), showing GUI.
```python
python human_play.py --weights path_to_model.model --width 8 --alpha_num 1200 --res_num 2 --show_GUI
```
Play with pure MCTS agent, with board size 11, win condition 5-in-a-row, number of playout 3000, showing game on terminal.
```python
python human_play.py --width 11 --number_in_row 5 --pure_num 3000 
```


### Update 2021.2.28: import from junxiaosong/AlphaZero_Gomoku
### Update 2021.3.15: re-implement Net utils for neural net utilities



