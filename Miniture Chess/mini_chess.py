import gymnasium as gym 
from gymnasium import spaces
import numpy as np
import time 
import copy

class MiniChessEnv(gym.Env):
    """
    A simplified 4x4 Mini Chess environment
    Pieces: 
    - White: Pawn (P), Rook (R), Knight (N), Queen (Q), King (K)
    - Black: pawn, rook, knight, queen, king (lowercase)
    Goal: Capture opponent's king
    """

    def __init__(self):
        self.board_size = 4
        # 0 = empty, 1=P, 2 = R, 4 = N, 4 = Q, 5 = K, and same for black but negative
        self.observation_space = spaces.Box(low=-6, high=6, shape=(4, 4), dtype=int)
        # Action: (from_row, from_col,  to_row, to_col)
        self.action_space = spaces.MultiDiscrete([4,4,4,4])
        self.reset()
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = self._initial_board()
        return self.board, {}
    ######################################################################################
    #               ALL MOVEMENT + LEGALITY FUNCTIONS
    ######################################################################################

    def _initial_board(self):
        board = np.zeros((4,4), dtype=int)

        return board
    def _get_obs(self):
        return { "board" : self.board.copy(),
                 "current_player" : self.current_player
            }
    
    def legal_moves(self, r, c):
        piece = self.board[r][c]
        if piece ==1:
            return self._pawn_moves(r,c)
        elif piece ==2 :
            return self._rook_moves(r,c)
        elif piece == 3:
            return self._knight_moves(r,c)
        elif piece == 4:
            return self._queen_moves(r,c)
        elif piece == 5:
            return self._king_moves(r,c)
        
        return []
    
    def _piece_color(self, piece):
        if piece > 0: return 1
        if piece < 0: return -1
        return 0
        
    def _pawn_moves(self, r, c):
        moves = []
        piece = self.board[c][r]
        if piece == 0:
            return moves
        color = self._piece_color(piece)

        forward = -1 if color == 1 else 1

        next_row = r + forward

        if 0 <= next_row < self.board_size and self.board[next_row][c] == 0:
            moves.append((next_row,c))

        # Diagonal Captures
        # it's an array not an interval
        for dc in [-1,1]:
            next_col = c + dc
            if 0 <= next_row < self.board_size and 0 <= next_col < self.board_size:
                target = self.board[next_row][next_col]
                if target != 0 and self._peice_color(target) != color:
                    moves.append((next_row, next_col))    
        return moves
    
    def _rook_moves(self, r,c):
        moves = []
        piece = self.board[r][c]
        color = self._piece_color(piece)

        if piece == 0:
            return moves
        
        directions = [(-1,0), (1,0), (0,-1), (0,1)]
        #delta_row/col
        for dr, dc in directions:
            #new_row/col
            nr, nc = r + dr, c + dc
            while 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                target = self.board[nr][nc]

                if target == 0:
                    moves.append((nr, nc))
                else:
                    if self._piece_color(target) != color:
                        moves.append((nr, nc))
                        # We've encountered another piece so that's all our options in this direction
                        break
                nr += dr
                nc += dc
        return moves
    
    def _queen_moves(self, r,c):
        moves = []
        piece = self.board[r][c]
        color = self._piece_color(piece)

        if piece == 0:
            return moves
        
        directions = [(-1,0), (1,0), (0,-1), (0,1), (1,1), (-1,-1), (1,-1), (-1,1)]

        for dr, dc in directions:
            nr = r + dr
            nc = c + dc
            while 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                target = self.board[nr][nc]
                if target == 0:
                    moves.append((nr, nc))
                else:
                    if self._piece_color(target) != color:
                        moves.append((nr, nc))
                        break
                nr += dr
                nc += dc 
        return moves

    def _knight_moves(self, r, c):
        piece = self.board[r][c]
        if piece == 0: 
            return []
        color = self._piece_color(piece)

        moves = []

        knight_jumps = [
            (-2, -1), (-2, 1),
            (-1, -2), (-1, 2),
            ( 1, -2), ( 1, 2),
            ( 2, -1), ( 2, 1)
        ]
        for dr, dc in knight_jumps:
            nr, nc = r+ dr, c+ dc
            if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                target = self.board[nr][nc]

                if target == 0:
                    moves.append((nr, nc))
                elif self._piece_color(target) != color:
                    moves.append((nr, nc))
        return moves

    def _king_moves(self, r, c):
        piece = self.board[r][c]
        if piece == 0: 
            return []
        color = self._piece_color(piece)
        moves = []

        knight_jumps = [
            (1, 0), (-1, 0),
            (0, 1), (0, -1),
            (1, 1), (1, -1),
            (-1, -1), (-1, 1)
        ]
        for dr, dc in knight_jumps:
            nr, nc = r+ dr, c+ dc

            if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                target = self.board[nr][nc]

                if target == 0:
                    moves.append((nr, nc))
                elif self._piece_color(target) != color:
                    moves.append((nr, nc))
        return moves

    def _compute_reward(self, fr, fc, tr, tc, target_piece, illegal=False):
        """
        fr, fc: from position
        tr, tc: to position
        target_piece: piece at destination before the move
        illegal: True if the move is illegal
        """
        if illegal:
            return -0.5
         
        reward = 0.0

        if target_piece !=0:
            reward += 0.1
        if abs(target_piece) == 5:
            reward = 1.0
        
        piece = self.board[tr][tc]
        if abs(piece) == 1:
            direction = -1 if piece > 0 else 1
            if tr == fr + direction:
                reward += 0.01
        return reward 



    def step(self, action):
        fr, fc, tr, tc = action
        piece = self.board[fr][fc]

        if piece == 0 or self._piece_color(piece) != self.current_player:
            reward = self._compute_reward(fr, fc, tr, tc, target_piece=0, illegal=True)
            return self._get_obs(), reward, False, False, {"illegal": True}
                
        legal = self.legal_moves(fr, fc)
        if (tr, tc) not in legal:
            reward = self._compute_reward(fr, fc, tr, tc, target_piece=0, illegal=True)
            return self._get_obs(), reward, False, False, {"illegal": True}
        
        target_piece = self.board[tr][tc]
        self.board[tr][tc] = piece
        self.board[fr][fc] = 0

        reward = self._compute_reward(fr, fc, tr, tc, target_piece)
        done = False

        done = abs(target_piece) == 5
        
        self.current_player *= -1

        return self.get_obs(), reward, done, False, {}
    
    def render(self):
        piece_map = {
            0: ".", 1: "P", 2:"R", 3:"N", 4:"Q", 5:"K",
            -1:"p", -2:"r", -3:"n", -4:"q", -5:"k"
        }
        print("Board:")
        for row in self.board:
            print(" ".join(piece_map[x] for x in row))
            print(f"Current player: {'White' if self.current_player==1 else 'Black'}\n")

#### TRAINING THE MODEL
env = MiniChessEnv()
state, _ = env.reset()
env.render()

done = False
while not done: 
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)
    env.render()
    done = terminated
