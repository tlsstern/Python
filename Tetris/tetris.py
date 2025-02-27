import pygame
import random
import copy

# Initialize pygame and font system
pygame.init()
pygame.font.init()

# -----------------------
# Game Constants
# -----------------------
CELL_SIZE = 30
COLS = 10
ROWS = 20
WIDTH = CELL_SIZE * COLS
HEIGHT = CELL_SIZE * ROWS
FPS = 60

# Colors
BLACK = (0, 0, 0)
COLORS = {
    'I': (0, 255, 255),
    'O': (255, 255, 0),
    'T': (128, 0, 128),
    'S': (0, 255, 0),
    'Z': (255, 0, 0),
    'J': (0, 0, 255),
    'L': (255, 165, 0)
}

# Tetromino base shapes (each is a list of (x, y) offsets)
TETROMINOS = {
    'I': [(0,1), (1,1), (2,1), (3,1)],
    'O': [(0,0), (1,0), (0,1), (1,1)],
    'T': [(1,0), (0,1), (1,1), (2,1)],
    'S': [(1,0), (2,0), (0,1), (1,1)],
    'Z': [(0,0), (1,0), (1,1), (2,1)],
    'J': [(0,0), (0,1), (1,1), (2,1)],
    'L': [(2,0), (0,1), (1,1), (2,1)]
}

# -----------------------
# Helper Functions
# -----------------------
def rotate_piece(piece, times):
    """Rotate a piece (list of (x,y) blocks) 90° clockwise 'times' times."""
    rotated = piece
    for _ in range(times % 4):
        rotated = [(y, -x) for (x, y) in rotated]
        # Normalize so that the minimum x and y become 0
        min_x = min(x for (x, y) in rotated)
        min_y = min(y for (x, y) in rotated)
        rotated = [(x - min_x, y - min_y) for (x, y) in rotated]
    return rotated

# -----------------------
# Game Classes
# -----------------------
class Piece:
    def __init__(self, shape):
        self.shape = shape
        self.rotation = 0
        self.blocks = rotate_piece(TETROMINOS[self.shape], self.rotation)
        self.x = COLS // 2 - 2  # Start roughly in the center
        self.y = 0

    def rotate(self):
        self.rotation = (self.rotation + 1) % 4
        self.blocks = rotate_piece(TETROMINOS[self.shape], self.rotation)

class TetrisGame:
    def __init__(self):
        # The board is a ROWS x COLS grid; each cell is None or a color tuple.
        self.board = [[None for _ in range(COLS)] for _ in range(ROWS)]
        self.current_piece = self.new_piece()
        self.score = 0
        self.game_over = False
        self.ai_target_move = None  # (target_rotation, target_x)

    def new_piece(self):
        shape = random.choice(list(TETROMINOS.keys()))
        return Piece(shape)

    def valid_position(self, piece, offset_x, offset_y):
        """Return True if the piece is in a valid position on the board."""
        for (x, y) in piece.blocks:
            new_x = piece.x + x + offset_x
            new_y = piece.y + y + offset_y
            if new_x < 0 or new_x >= COLS or new_y >= ROWS:
                return False
            if new_y >= 0 and self.board[new_y][new_x] is not None:
                return False
        return True

    def lock_piece(self, piece):
        """Lock the piece into the board and clear any full lines."""
        for (x, y) in piece.blocks:
            bx = piece.x + x
            by = piece.y + y
            if by < 0:
                self.game_over = True
            else:
                self.board[by][bx] = COLORS[piece.shape]
        self.clear_lines()
        self.current_piece = self.new_piece()
        self.ai_target_move = None  # Reset AI move for the new piece

    def clear_lines(self):
        new_board = [row for row in self.board if any(cell is None for cell in row)]
        lines_cleared = ROWS - len(new_board)
        # Increase score based on the number of lines cleared.
        self.score += lines_cleared * 100
        for _ in range(lines_cleared):
            new_board.insert(0, [None for _ in range(COLS)])
        self.board = new_board

    def step(self):
        """Advance the game by one gravity step."""
        if self.valid_position(self.current_piece, 0, 1):
            self.current_piece.y += 1
        else:
            self.lock_piece(self.current_piece)

    def ai_move(self):
        """Have the AI gradually move the current piece toward its target placement."""
        if self.ai_target_move is None:
            move = get_best_move(self)
            if move is not None:
                self.ai_target_move = move
            else:
                return  # No valid move found; just drop the piece

        target_rotation, target_x = self.ai_target_move

        # Rotate if needed
        if self.current_piece.rotation != target_rotation:
            new_piece = copy.deepcopy(self.current_piece)
            new_piece.rotate()
            if self.valid_position(new_piece, 0, 0):
                self.current_piece.rotate()
            return

        # Move horizontally toward target_x
        if self.current_piece.x < target_x:
            if self.valid_position(self.current_piece, 1, 0):
                self.current_piece.x += 1
            return
        elif self.current_piece.x > target_x:
            if self.valid_position(self.current_piece, -1, 0):
                self.current_piece.x -= 1
            return
        # When aligned, gravity (in step()) will drop the piece

# -----------------------
# AI Helper Functions
# -----------------------
def valid_position_board(board, piece, offset_x, offset_y):
    """Check if piece is in a valid position on the given board."""
    for (x, y) in piece.blocks:
        new_x = piece.x + x + offset_x
        new_y = piece.y + y + offset_y
        if new_x < 0 or new_x >= COLS or new_y >= ROWS:
            return False
        if new_y >= 0 and board[new_y][new_x] is not None:
            return False
    return True

def clear_lines_sim(board):
    """Simulate clearing complete lines on a board copy."""
    new_board = [row for row in board if any(cell is None for cell in row)]
    lines_cleared = ROWS - len(new_board)
    for _ in range(lines_cleared):
        new_board.insert(0, [None for _ in range(COLS)])
    return new_board

def column_height(board, col):
    for i in range(ROWS):
        if board[i][col] is not None:
            return ROWS - i
    return 0

def count_complete_lines(board):
    count = 0
    for row in board:
        if all(cell is not None for cell in row):
            count += 1
    return count

def count_holes(board):
    holes = 0
    for col in range(COLS):
        block_found = False
        for row in range(ROWS):
            if board[row][col] is not None:
                block_found = True
            elif block_found:
                holes += 1
    return holes

def calculate_bumpiness(board):
    heights = [column_height(board, col) for col in range(COLS)]
    bumpiness = 0
    for i in range(len(heights) - 1):
        bumpiness += abs(heights[i] - heights[i+1])
    return bumpiness

def evaluate_board(board):
    """Return a heuristic score for the board state.
    Lower aggregate height, fewer holes, and less bumpiness are better,
    while more complete lines are good."""
    aggregate_height = sum(column_height(board, col) for col in range(COLS))
    complete_lines = count_complete_lines(board)
    holes = count_holes(board)
    bumpiness = calculate_bumpiness(board)
    score = (-0.510066 * aggregate_height +
             0.760666 * complete_lines -
             0.35663 * holes -
             0.184483 * bumpiness)
    return score

def simulate_drop(board, piece, x, rotation):
    """Simulate dropping a piece (with given rotation and x position)
    onto the board and return the new board state after placement."""
    sim_piece = Piece(piece.shape)
    sim_piece.rotation = rotation
    sim_piece.blocks = rotate_piece(TETROMINOS[piece.shape], rotation)
    sim_piece.x = x
    sim_piece.y = piece.y
    while valid_position_board(board, sim_piece, 0, 1):
        sim_piece.y += 1
    board_copy = copy.deepcopy(board)
    for (x_offset, y_offset) in sim_piece.blocks:
        bx = sim_piece.x + x_offset
        by = sim_piece.y + y_offset
        if 0 <= by < ROWS and 0 <= bx < COLS:
            board_copy[by][bx] = COLORS[sim_piece.shape]
    board_copy = clear_lines_sim(board_copy)
    return board_copy

def get_best_move(game):
    """Examine all possible rotations and x positions for the current piece,
    simulate dropping it, and pick the move with the best heuristic score."""
    best_score = -float('inf')
    best_move = None
    piece = game.current_piece
    # Try all 4 rotations
    for rotation in range(4):
        rotated_blocks = rotate_piece(TETROMINOS[piece.shape], rotation)
        # Determine horizontal boundaries for the piece
        min_x = min(x for (x, y) in rotated_blocks)
        max_x = max(x for (x, y) in rotated_blocks)
        for x in range(-min_x, COLS - max_x):
            temp_piece = Piece(piece.shape)
            temp_piece.rotation = rotation
            temp_piece.blocks = rotated_blocks
            temp_piece.x = x
            temp_piece.y = piece.y
            if not valid_position_board(game.board, temp_piece, 0, 0):
                continue
            board_after = simulate_drop(game.board, temp_piece, x, rotation)
            score = evaluate_board(board_after)
            if score > best_score:
                best_score = score
                best_move = (rotation, x)
    return best_move

# -----------------------
# Rendering Functions
# -----------------------
def draw_board(screen, board):
    """Draw the board without grid borders."""
    for y, row in enumerate(board):
        for x, cell in enumerate(row):
            if cell is not None:
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, cell, rect)

def draw_piece(screen, piece):
    """Draw the current piece without borders."""
    for (x, y) in piece.blocks:
        bx = (piece.x + x) * CELL_SIZE
        by = (piece.y + y) * CELL_SIZE
        rect = pygame.Rect(bx, by, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, COLORS[piece.shape], rect)

def draw_score(screen, score, font):
    """Render the score on the screen."""
    score_text = font.render(f"Score: {score}", True, (255, 255, 255))
    screen.blit(score_text, (10, 10))

# -----------------------
# Main Game Loop
# -----------------------
def main():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Self-Playing Tetris")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 24)

    game = TetrisGame()
    drop_timer = 0
    normal_drop_interval = 200   # Normal drop interval in milliseconds
    fast_drop_interval = 10      # Fast drop interval when down arrow is pressed

    down_pressed = False  # Flag to track if down arrow is held

    running = True
    while running:
        dt = clock.tick(FPS)
        drop_timer += dt

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_DOWN:
                    down_pressed = True
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_DOWN:
                    down_pressed = False

        # Choose drop interval based on whether the down arrow is pressed.
        current_drop_interval = fast_drop_interval if down_pressed else normal_drop_interval

        if not game.game_over:
            # Let the AI move the piece toward its target placement.
            game.ai_move()
            # Drop the piece at the current interval.
            if drop_timer > current_drop_interval:
                game.step()
                drop_timer = 0

        # Render the game
        screen.fill(BLACK)
        draw_board(screen, game.board)
        draw_piece(screen, game.current_piece)
        draw_score(screen, game.score, font)
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
