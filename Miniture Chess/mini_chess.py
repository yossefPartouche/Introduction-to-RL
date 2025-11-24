import gymnasium as gym 
from gymnasium import spaces
import numpy as np
import time 
import pygame

class MiniChessEnv(gym.Env):
    """
    A simplified 4x4 Mini Chess environment
    Pieces: 
    - White: Pawn (P), Rook (R), Knight (N), Queen (Q), King (K)
    - Black: pawn, rook, knight, queen, king (lowercase)
    Goal: Capture opponent's king
    """
    metadata = {"render_mode": ["human", "rgb_array"], "render_fps": 4} # this is for the improved rendering
    
    def __init__(self, render_mode=None):
        super(MiniChessEnv, self).__init__()
        self.board_size = 4
        self.current_player = 1
        self.observation_space = spaces.Box(low=-6, high=6, shape=(4,4), dtype=int)
        self.action_space = spaces.MultiDiscrete([4,4,4,4])

        # PYGAME SETUP
        self.render_mode = render_mode
        self.window_size = 512
        self.window = None
        self.clock = None
        self.cell_size = self.window_size // self.board_size

        self.reset()

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            self._render_frame()
        
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        canvas = pygame.Surface((self.window_size, self.window_size))

        # Colors
        WHITE_SQ = (235, 236, 208) # Classic Chess.com light square
        DARK_SQ = (119, 149, 86)   # Classic Chess.com dark square

        # 1. Draw the Board
        for r in range(self.board_size):
            for c in range(self.board_size):
                color = WHITE_SQ if (r +c) % 2 == 0 else DARK_SQ
                pygame.draw.rect(
                    canvas, 
                        color,
                    pygame.Rect(
                        c * self.cell_size, 
                        r *self.cell_size, 
                        self.cell_size, 
                        self.cell_size
                    ),
                )
        system_fonts = pygame.font.get_fonts()

        font_name = "arial"
        # Pick the best font based on OS
        if "applesymbols" in system_fonts:
            font_name = "applesymbols"  # <--- The Fix for Mac
        elif "segoeuisymbol" in system_fonts:
            font_name = "segoeuisymbol" # The Fix for Windows
        elif "dejavusans" in system_fonts:
            font_name = "dejavusans"    # The Fix for Linux
        # 2. Draw pieces
        # Using Pygame fonts to render Unicode chess pieces
        font = pygame.font.SysFont(font_name, int(self.cell_size * 0.8))
        symbols = {
            1: "♙", 2: "♖", 3: "♘", 4: "♕", 5: "♔",
            -1: "♟", -2: "♜", -3: "♞", -4: "♛", -5: "♚"
        }
        for r in range(self.board_size):
            for c in range(self.board_size):
                piece = self.board[r][c]
                if piece == 0: continue

                txt = symbols[abs(piece)]

                if piece > 0:
                    text_color = (255, 255, 255)
                    outline_color = (0,0,0)
                else:
                    text_color = (0,0,0)
                    outline_color = (255, 255, 255)

                # render text
                text_surface = font.render(txt, True, text_color)
                text_rect = text_surface.get_rect(center=(
                    c * self.cell_size + self.cell_size // 2,
                    r * self.cell_size + self.cell_size // 2
                ))
                canvas.blit(text_surface, text_rect)
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

    

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        # White pieces (Rows 2 and 3)
        self.board[3] = [2, 3, 4, 5] # R, N, Q, K
        self.board[2] = [1, 1, 1, 1] # Pawns
        
        # Black pieces (Rows 0 and 1)
        self.board[0] = [-2, -3, -4, -5] # r, n, q, k
        self.board[1] = [-1, -1, -1, -1] # pawns
        self.current_player = 1  # White starts
        return self._get_obs(), {}
    ######################################################################################
    #               ALL MOVEMENT + LEGALITY FUNCTIONS
    ######################################################################################
    def _get_obs(self):
        return { "board" : self.board.copy(),
                 "current_player" : self.current_player
            }
    
    def _piece_color(self, piece):
        if piece > 0: return 1
        if piece < 0: return -1
        return 0
    
    def legal_moves(self, r, c):
        piece = self.board[r][c]
        if piece == 0: return []

        abs_piece = abs(piece)

        if abs_piece ==1: return self._pawn_moves(r,c)
        if abs_piece ==2: return self._rook_moves(r,c)
        if abs_piece == 3: return self._knight_moves(r,c)
        if abs_piece == 4: return self._queen_moves(r,c)
        if abs_piece == 5: return self._king_moves(r,c)
        return []

        
    def _pawn_moves(self, r, c):
        moves = []
        piece = self.board[r][c]
        color = self._piece_color(piece)

        forward = -1 if color == 1 else 1
        next_row = r + forward

        if piece == 0:
            return moves

        if 0 <= next_row < self.board_size:
            if self.board[next_row][c] == 0:
                moves.append((next_row,c))

        # Diagonal Captures
        # it's an array not an interval
        for dc in [-1,1]:
            next_col = c + dc
            if 0 <= next_row < self.board_size and 0 <= next_col < self.board_size:
                target = self.board[next_row][next_col]
                if target != 0 and self._piece_color(target) != color:
                    moves.append((next_row, next_col))    
        return moves
    
    def _rook_moves(self, r,c):
        moves = []
        piece = self.board[r][c]
        color = self._piece_color(piece)
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
                        
                        break # Blocked by piece (friend or foe)
                nr += dr
                nc += dc
        return moves
    
    def _queen_moves(self, r,c):
        moves = []
        piece = self.board[r][c]
        color = self._piece_color(piece)
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
        moves = []
        piece = self.board[r][c]
        color = self._piece_color(piece)

        jumps = [
            (-2, -1), (-2, 1),
            (-1, -2), (-1, 2),
            ( 1, -2), ( 1, 2),
            ( 2, -1), ( 2, 1)
        ]
        for dr, dc in jumps:
            nr, nc = r+ dr, c+ dc
            if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                target = self.board[nr][nc]

                if target == 0 or self._piece_color(target)!=color:
                    moves.append((nr, nc))
        return moves

    def _king_moves(self, r, c):
        piece = self.board[r][c]
        color = self._piece_color(piece)
        moves = []

        deltas = [
            (1, 0), (-1, 0),
            (0, 1), (0, -1),
            (1, 1), (1, -1),
            (-1, -1), (-1, 1)
        ]
        for dr, dc in deltas:
            nr, nc = r+ dr, c+ dc
            if 0 <= nr < self.board_size and 0 <= nc < self.board_size:
                target = self.board[nr][nc]
                if target == 0 or self._piece_color(target) != color:
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
            return -10
         
        reward = 0.0

        if target_piece !=0:
            reward += 1.0
        if abs(target_piece) == 5:
            reward = 100.0
        
        piece = self.board[tr][tc]
        if abs(piece) == 1:
            reward += 0.1
        return reward 



    def step(self, action):
        fr, fc, tr, tc = action
        piece = self.board[fr][fc]

        if piece == 0 or self._piece_color(piece) != self.current_player:
            return self._get_obs(), -10, False, False, {"illegal": True, "reason": "No piece/Wrong Color"}
                
        legal = self.legal_moves(fr, fc)
        if (tr, tc) not in legal:
            return self._get_obs(), -10, False, False, {"illegal": True, "reason": "Invalid Move Geometry"}
        
        target_piece = self.board[tr][tc]
        self.board[tr][tc] = piece
        self.board[fr][fc] = 0

        reward = self._compute_reward(fr, fc, tr, tc, target_piece)

        done = abs(target_piece) == 5

        self.current_player *= -1

        return self._get_obs(), reward, done, False, {}
    """
    ##### BASIC VISUALISATION
    def __init__(self):
        

        super(MiniChessEnv, self).__init__()
        self.board_size = 4
        self.current_player = 1
        # 0 = empty, 1=P, 2 = R, 4 = N, 4 = Q, 5 = K, and same for black but negative
        self.observation_space = spaces.Box(low=-6, high=6, shape=(4, 4), dtype=int)
        # Action: (from_row, from_col,  to_row, to_col)
        self.action_space = spaces.MultiDiscrete([4,4,4,4])
        self.reset()

    def render(self):
        piece_map = {
            0: ".", 1: "P", 2:"R", 3:"N", 4:"Q", 5:"K",
            -1:"p", -2:"r", -3:"n", -4:"q", -5:"k"
        }
        print("\n   0  1  2  3")
        for r in range(self.board_size):
            row_str = f"{r} "
            for c in range(self.board_size):
                row_str += piece_map[self.board[r][c]]
            print(row_str)
        print(f"Current Turn: {'White (+)' if self.current_player==1 else 'Black (-)'}")
        print("-" * 20)
    """

# --- RUNNER SCRIPT ---

env = MiniChessEnv(render_mode="human")
obs, _ = env.reset()
env.render()

done = False
steps = 0 
max_steps = 50 # Stops after min(50, game_steps)

print("Starting Random Game...")
time.sleep(1)

while not done and steps < max_steps:
    steps += 1

    # 1. find all possible legal moves for the current player
    all_legal_moves = []
    for r in range(env.board_size):
        for c in range(env.board_size):
            piece = env.board[r][c]
            if piece != 0 and env._piece_color(piece) == env.current_player:
                destinations = env.legal_moves(r,c)
                for tr, tc in destinations:
                    all_legal_moves.append((r,c,tr,tc))

    if not all_legal_moves: 
        print(f"No legal moves for {'White' if env.current_player == 1 else 'Black'}! Game Over.")
        break

    # 2. check for Stalemate / No moves left
    idx = np.random.randint(len(all_legal_moves))
    action = all_legal_moves[idx]
    
    # 3. Pick a random move
    print(f"Step {steps}: Player {env.current_player} moves {action}")

    # 4. Step environment
    obs, reward, done, truncated, info = env.step(action)
    
    env.render()
    time.sleep(1) # Slow down so you can see what happens

if done:
    print("Checkmate / King Captured!")
elif steps >= max_steps:
    print("Draw (Max steps reached)")

env.close()