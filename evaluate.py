from game import Board, Game
from MCTS import MCTSPlayer as MCTS_Pure
from MCTS_alpha import MCTSPlayerAlpha as MCTSPlayer
import argparse
from pathlib import Path
from Net_util_pytorch import PolicyValueNet
from collections import defaultdict
import matplotlib.pyplot as plt

# evaluate the model
def evaluate(opt):
    n_in_row = opt.num_in_row
    num_player = opt.num_player
    num_round = opt.num_round
    board_size = opt.width
    width, height = board_size, board_size
    player1 = opt.player1
    player2 = opt.player2
    weights1 = opt.weights1
    weights2 = opt.weights2
    c_puct1 = 5
    c_puct2 = 5
    res_num1 = opt.res_num1
    res_num2 = opt.res_num2
    n_playout1 = opt.n_playout1
    n_playout2 = opt.n_playout2
    if (num_round <= 0):
        raise Exception(
            'number of round can not be less than {}'.format(num_round))

    # create board and game
    board = Board(width=width, height=height, n_in_row=n_in_row)
    game = Game(board)
    print(
        "Board width* height: {}*{}, win condition: {}, round of test(total round*2*3 games): {}"
        .format(width, height, n_in_row, num_round))

    # create players
    print("Create Player 1")
    Player1_1 = create_player(width, height, player1, weights1, c_puct1,
                              res_num1, n_playout1)
    Player1_2 = create_player(width, height, player1, weights1, c_puct1,
                              res_num1, n_playout1)
    print("Create Player 2")
    Player2_1 = create_player(width, height, player2, weights2, c_puct2,
                              res_num2, n_playout2)
    Player2_2 = create_player(width, height, player2, weights2, c_puct2,
                              res_num2, n_playout2)

    # begin test
    print("1 Player1 vs 2 Player2:")
    win_cnt1, win_score1 = round_play_test(game, num_round, Player1_1,
                                           Player2_1, Player2_2)
    print("Player1 score: {}".format(win_score1))
    print("2 Player1 vs 1 Player2:")
    win_cnt2, win_score2 = round_play_test(game, num_round, Player2_1,
                                           Player1_1, Player1_2)
    print("Player2 score: {}".format(win_score2))

    if (abs(win_score1 - win_score2) < 1.0):
        print("Player1 and Player2's performance are close")
    elif (win_score1 < win_score2):
        print("Player2 better than Player1")
    else:
        print("Player1 better than Player2")


def round_play_test(game, num_round, player1, player2, player3):

    win_cnt = defaultdict(int)
    for i in range(3 * num_round):
        winner = game.start_play(player1,
                                 player2,
                                 player3,
                                 start_player=i % 3,
                                 is_shown=0,
                                 show_end=0)
        win_cnt[winner] += 1
    win_score = (3.0 * win_cnt[0] + 1.0 * win_cnt[-1]) / num_round
    print("score: {}, win: {}, lose 1: {}, lose 2: {}, tie:{}".format(
        win_score, win_cnt[0], win_cnt[1], win_cnt[2], win_cnt[-1]))
    return win_cnt, win_score


def create_player(width, height, player, weights, c_puct, res_num, n_playout):
    try:
        if player == 'alpha_mcts':
            if weights == '':
                raise ValueError("need weight to initial alpha MCTS player")
            model_file = Path(weights)  #'best_policy_8_8_5.model'
            best_policy = PolicyValueNet(width,
                                         height,
                                         res_num=res_num,
                                         model_file=model_file)
            current_player = MCTSPlayer(
                policy_value_fn=best_policy.policy_value_fn,
                c_puct=c_puct,
                n_playout=n_playout)
            print("player {}, res num: {}, n_playout: {}".format(
                current_player, res_num, n_playout))
        elif player == 'pure_mcts':
            current_player = MCTS_Pure(c_puct=c_puct, n_playout=n_playout)
            print("player {}, n_playout: {}".format(current_player, n_playout))
        elif player == 'min_max':
            raise ValueError("don't finish yet")
        else:
            raise ValueError(
                "should choose a kind of player from pure_mcts, alpha_mcts, min_max"
            )
    except ValueError as e:
        print(repr(e))
    else:
        return current_player

    # # training data analysis
    # parser.add_argument('--pharse', nargs='?', const=True, default=False, help='pharse training data or not, call default True')
    # parser.add_argument('--weights


# plot the training data file
def pharsing():
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # evaluate
    parser.add_argument('--eval',
                        nargs='?',
                        const=True,
                        default=False,
                        help='evaluate players or not, call default True')
    parser.add_argument('--width',
                        type=int,
                        default=6,
                        help='width of board, init 6')
    parser.add_argument('--num_in_row',
                        '-n',
                        type=int,
                        default=4,
                        help='win condition, init 4')
    parser.add_argument(
        '--player1',
        '-p1',
        type=str,
        default='',
        help='Agent 1 type, init empty, options: pure_mcts, alpha_mcts, min_max'
    )
    parser.add_argument(
        '--player2',
        '-p2',
        type=str,
        default='',
        help='Agent 2 type, init empty, options: pure_mcts, alpha_mcts, min_max'
    )
    parser.add_argument(
        '--weights1',
        '-w1',
        type=str,
        default='',
        help=
        'Agent 1 weights path, init empty, empty lead to pure MCTS, weight lead to Alpha MCTS'
    )
    parser.add_argument(
        '--weights2',
        '-w2',
        type=str,
        default='',
        help=
        'Agent 2 weights path, init empty, empty lead to pure MCTS, weight lead to Alpha MCTS'
    )
    parser.add_argument('--res_num1',
                        type=int,
                        default=0,
                        help='player1 res block num, init 0')
    parser.add_argument('--res_num2',
                        type=int,
                        default=0,
                        help='player2 res block num, init 0')
    parser.add_argument('--n_playout1',
                        type=int,
                        default=1000,
                        help='play out numbers of player1, default 1000')
    parser.add_argument('--n_playout2',
                        type=int,
                        default=1000,
                        help='play out numbers of player2, default 1000')
    parser.add_argument('--num_player',
                        '-np',
                        type=int,
                        default=3,
                        help='number of players, init 3')
    parser.add_argument('--num_round',
                        '-nd',
                        type=int,
                        default=3,
                        help='number of rounds to play, init 3*num of player')
    # training data analysis
    parser.add_argument('--pharse',
                        nargs='?',
                        const=True,
                        default=False,
                        help='pharse training data or not, call default True')
    parser.add_argument(
        '--weights',
        '-w',
        type=str,
        default='',
        help=
        'take multiple weights paths, init empty, pharse several continuous path and analyze together'
    )

    opt = parser.parse_args()

    if opt.eval:
        evaluate(opt)
    if opt.pharse:
        pharsing(opt)