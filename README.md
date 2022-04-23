# AlphaZero-Gobang-3players
## Unique works:
~ This is an implementation of the AlphaZero/pure MCTS algorithm for playing the simple board game Gobang for 3 players.  
~ Control the player number(2 and 3 tested) and neural net structure(number of residual block)  
~ Design a evaluation system testing agent's performance 

## Future goals:
 
~ Adding min-max agent in agent operations  
~ Choosing agent in a more elegent way(argpharse and GUI)  

## References:   
~ Import 2-player source code from: https://github.com/junxiaosong/AlphaZero_Gomoku  
~ Combine GUI from: https://github.com/zouyih/AlphaZero_Gomoku-tensorflow  
~ AlphaZero: Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm  
~ AlphaGo Zero: Mastering the game of Go without human knowledge  

## Code file usage
### training
using pharse to control train.py.
```
python train.py --weights path_to_model.model --save_dir models_name --width 8 --n_playout 1200 --init_playout 3000 --resume --init_batch 2700 --max_batch 5000 --res_num 2 --use_gpu
```
Available argparse are listed below
```
python train.py --help                   
usage: train.py [-h] [--weights WEIGHTS] [--save_dir SAVE_DIR] [--number_player NUMBER_PLAYER] [--width WIDTH] [--number_in_row NUMBER_IN_ROW] [--n_playout N_PLAYOUT]
                [--res_num RES_NUM] [--check_freq CHECK_FREQ] [--init_playout INIT_PLAYOUT] [--max_playout MAX_PLAYOUT] [--resume [RESUME]] [--init_batch INIT_BATCH]
                [--max_batch MAX_BATCH] [--use_gpu [USE_GPU]]

optional arguments:
  -h, --help            show this help message and exit
  --weights WEIGHTS, -w WEIGHTS
                        initial weights path, init empty
  --save_dir SAVE_DIR   save to models/save_dir+desc/xxxx, desc include width, num in row, init model
  --number_player NUMBER_PLAYER, -np NUMBER_PLAYER
                        number of players, init 3
  --width WIDTH         width of board, init 6
  --number_in_row NUMBER_IN_ROW, -n NUMBER_IN_ROW
                        win condition, init 4
  --n_playout N_PLAYOUT
                        Alpha MCTS playout num, init 800
  --res_num RES_NUM     res block num, init 0
  --check_freq CHECK_FREQ
                        performance check freq, init 50
  --init_playout INIT_PLAYOUT
                        initial pure MCTS playout, init 1000
  --max_playout MAX_PLAYOUT
                        max pure MCTS playout, init 9000
  --resume [RESUME]     resume most recent training, init False
  --init_batch INIT_BATCH
                        initial batch number, init 0
  --max_batch MAX_BATCH
                        max batch number, init 1500
  --use_gpu [USE_GPU]   using gpu, init False

```
### human play
Using human_play.py to play with agents. Play with AlphaZero agent, with board size 8, number of playout 1200, 2 residual blocks (should match model file), showing GUI.
```
python human_play.py --weights path_to_model.model --width 8 --alpha_num 1200 --res_num 2 --show_GUI
```
Play with pure MCTS agent, with board size 11, win condition 5-in-a-row, number of playout 3000, showing game on terminal.
```
python human_play.py --width 11 --number_in_row 5 --pure_num 3000 
```
Play with pure MCTS agent
```
python human_play.py --min_max
```
Available argparse are listed below
```
python human_play.py --help
usage: human_play.py [-h] [--weights WEIGHTS] [--number_player NUMBER_PLAYER] [--width WIDTH] [--number_in_row NUMBER_IN_ROW] [--start START] [--pure_num PURE_NUM]
                     [--alpha_num ALPHA_NUM] [--show_GUI [SHOW_GUI]] [--res_num RES_NUM] [--min_max MIN_MAX]

optional arguments:
  -h, --help            show this help message and exit
  --weights WEIGHTS, -w WEIGHTS
                        initial weights path, init empty, empty lead to pure MCTS, weight lead to Alpha MCTS
  --number_player NUMBER_PLAYER, -np NUMBER_PLAYER
                        number of players, init 3
  --width WIDTH         width of board, init 6
  --number_in_row NUMBER_IN_ROW, -n NUMBER_IN_ROW
                        win condition, init 4
  --start START, -st START
                        start number of players
  --pure_num PURE_NUM   play out numbers of pure MCTS, default 2000
  --alpha_num ALPHA_NUM
                        play out numbers of Alpha MCTS, default 1000
  --show_GUI [SHOW_GUI]
                        draw GUI or not, default True
  --res_num RES_NUM     res block num, init 0
  --min_max MIN_MAX     play with min_max agent

```
### evaluate and analyze
Using evaluate.py, and use arguments to control the function.
evaluate two model
```
python evaluate.py --eval --width 8 -p1 alpha_mcts -p2 alpha_mcts -w1 models/models_0305_0035_continue1300_8_8_4/best_policy_750.model -w2 models/models_03091448_continue2700_8_8_4/best_policy_4350.model -nd 2
```
read and analyze training information
```
python evaluate.py --pharse -ws models/models_0315_2042_playout800_res0_8_8_4
```
read and analyze more continuous training sections
```
python evaluate.py --pharse -ws models/models_0304_0041_8_8_4/,models/models_0305_0035_continue1300_8_8_4/,models/models_03091448_continue2700_8_8_4

```
Available argparse are listed below
```
python evaluate.py --help
usage: evaluate.py [-h] [--eval [EVAL]] [--width WIDTH] [--num_in_row NUM_IN_ROW] [--player1 PLAYER1] [--player2 PLAYER2] [--weights1 WEIGHTS1] [--weights2 WEIGHTS2]
                   [--res_num1 RES_NUM1] [--res_num2 RES_NUM2] [--n_playout1 N_PLAYOUT1] [--n_playout2 N_PLAYOUT2] [--num_player NUM_PLAYER] [--num_round NUM_ROUND]
                   [--analyze [ANALYZE]] [--weights WEIGHTS]

optional arguments:
  -h, --help            show this help message and exit
  --eval [EVAL]         evaluate players or not, call default True
  --width WIDTH         width of board, init 6
  --num_in_row NUM_IN_ROW, -n NUM_IN_ROW
                        win condition, init 4
  --player1 PLAYER1, -p1 PLAYER1
                        Agent 1 type, init empty, options: pure_mcts, alpha_mcts, min_max
  --player2 PLAYER2, -p2 PLAYER2
                        Agent 2 type, init empty, options: pure_mcts, alpha_mcts, min_max
  --weights1 WEIGHTS1, -w1 WEIGHTS1
                        Agent 1 weights path, init empty, empty lead to pure MCTS, weight lead to Alpha MCTS
  --weights2 WEIGHTS2, -w2 WEIGHTS2
                        Agent 2 weights path, init empty, empty lead to pure MCTS, weight lead to Alpha MCTS
  --res_num1 RES_NUM1   player1 res block num, init 0
  --res_num2 RES_NUM2   player2 res block num, init 0
  --n_playout1 N_PLAYOUT1, -n1 N_PLAYOUT1
                        play out numbers of player1, default 1000
  --n_playout2 N_PLAYOUT2, -n2 N_PLAYOUT2
                        play out numbers of player2, default 1000
  --num_player NUM_PLAYER, -np NUM_PLAYER
                        number of players, init 3
  --num_round NUM_ROUND, -nd NUM_ROUND
                        number of rounds to play, init 3*num of player
  --analyze [ANALYZE]     analyze training data or not, call default True
  --weights WEIGHTS, -ws WEIGHTS
                        take multiple weights paths, init empty, pharse several continuous path and analyze together
```
## Updates
Update 2022.2.28: import from junxiaosong/AlphaZero_Gomoku  
Update 2022.3.03: 3-player mode  
Update 2022.3.04: res block available  
Update 2022.3.05: adding GUI  
Update 2022.3.12: re-implement pure MCTS agent  
Update 2022.3.13: re-implement alpha MCTS agent  
Update 2022.3.15: re-implement Net utils for neural net utilities  
Update 2022.3.18: adding evaluate.py to compare the performance of two kinds of agent, and analysis of training section data recorded during training  



