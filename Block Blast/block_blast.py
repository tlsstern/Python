import pygame
import sys
import random

# Constants
BOARD_SIZE = 10
CELL_SIZE = 40
BOARD_WIDTH = BOARD_SIZE * CELL_SIZE  # 400 px
BOARD_HEIGHT = BOARD_SIZE * CELL_SIZE  # 400 px
PIECE_AREA_HEIGHT = 150
SCORE_AREA_HEIGHT = 50
WINDOW_WIDTH = BOARD_WIDTH
WINDOW_HEIGHT = BOARD_HEIGHT + PIECE_AREA_HEIGHT + SCORE_AREA_HEIGHT  # 600 px total

# Dark mode colors
DARK_BG = (30, 30, 30)
BOARD_GRID_COLOR = (70, 70, 70)
TEXT_COLOR = (255, 255, 255)
SCORE_AREA_BG = (50, 50, 50)

# Block colors (kept bright for contrast)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
CYAN = (0, 255, 255)
MAGENTA = (255, 0, 255)

# Define block shapes and their colors
SHAPES = [
    {"shape": [(0, 0)], "color": RED},
    {"shape": [(0, 0), (1, 0)], "color": BLUE},
    {"shape": [(0, 0), (0, 1)], "color": GREEN},
    {"shape": [(0, 0), (1, 0), (2, 0)], "color": PURPLE},
    {"shape": [(0, 0), (0, 1), (0, 2)], "color": ORANGE},
    {"shape": [(0, 0), (1, 0), (0, 1), (1, 1)], "color": CYAN},
    {"shape": [(0, 0), (0, 1), (1, 1)], "color": MAGENTA},
]

# Slider gauge constants
GAUGE_WIDTH = 200
GAUGE_HEIGHT = 20


def init_board():
    """Create an empty BOARD_SIZE x BOARD_SIZE board."""
    return [[None for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]


def generate_pieces():
    """Generate a list of three random pieces."""
    pieces = []
    for _ in range(3):
        piece = random.choice(SHAPES)
        # Create a copy so the original shape isn't modified
        piece_copy = {"shape": [tuple(coord) for coord in piece["shape"]], "color": piece["color"]}
        pieces.append(piece_copy)
    return pieces


def can_place_piece(piece, board, board_x, board_y):
    """Check if a piece can be placed on the board at (board_x, board_y)."""
    for dx, dy in piece["shape"]:
        x = board_x + dx
        y = board_y + dy
        if x < 0 or x >= BOARD_SIZE or y < 0 or y >= BOARD_SIZE:
            return False
        if board[y][x] is not None:
            return False
    return True


def place_piece(piece, board, board_x, board_y):
    """Place the piece on the board."""
    for dx, dy in piece["shape"]:
        x = board_x + dx
        y = board_y + dy
        board[y][x] = piece["color"]


def check_clear(board):
    """Clear any full rows or columns and return the number of blocks cleared."""
    cleared = 0
    rows_to_clear = []
    cols_to_clear = []

    # Check full rows.
    for i in range(BOARD_SIZE):
        if all(board[i][j] is not None for j in range(BOARD_SIZE)):
            rows_to_clear.append(i)

    # Check full columns.
    for j in range(BOARD_SIZE):
        if all(board[i][j] is not None for i in range(BOARD_SIZE)):
            cols_to_clear.append(j)

    # Clear rows.
    for i in rows_to_clear:
        for j in range(BOARD_SIZE):
            if board[i][j] is not None:
                board[i][j] = None
                cleared += 1

    # Clear columns (avoiding double-clearing cells).
    for j in cols_to_clear:
        for i in range(BOARD_SIZE):
            if board[i][j] is not None:
                board[i][j] = None
                cleared += 1
    return cleared


def draw_board(screen, board):
    """Draw the board grid and any placed blocks."""
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, BOARD_GRID_COLOR, rect, 1)
            if board[i][j]:
                pygame.draw.rect(screen, board[i][j], rect)


def draw_piece_area(screen, pieces):
    """Draw available pieces in the piece area (without extra outlines)."""
    margin = 10
    x_offset = margin
    y_offset = BOARD_HEIGHT + 10
    for piece in pieces:
        for dx, dy in piece["shape"]:
            block_rect = pygame.Rect(x_offset + dx * CELL_SIZE, y_offset + dy * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, piece["color"], block_rect)
        # Advance the x offset based on piece width.
        max_x = max(dx for dx, dy in piece["shape"])
        width = (max_x + 1) * CELL_SIZE
        x_offset += width + margin


def draw_info(screen, score, speed, font):
    """Draw score and speed info, plus a slider gauge with a draggable knob."""
    # Draw the info area background.
    info_area_rect = pygame.Rect(0, BOARD_HEIGHT + PIECE_AREA_HEIGHT, WINDOW_WIDTH, SCORE_AREA_HEIGHT)
    pygame.draw.rect(screen, SCORE_AREA_BG, info_area_rect)

    # Render text showing current score and speed.
    info_text = font.render(f"Score: {score}    Speed: {speed}", True, TEXT_COLOR)
    text_x = (WINDOW_WIDTH - info_text.get_width()) // 2
    text_y = BOARD_HEIGHT + PIECE_AREA_HEIGHT + 5
    screen.blit(info_text, (text_x, text_y))

    # Draw the slider gauge.
    gauge_x = (WINDOW_WIDTH - GAUGE_WIDTH) // 2
    gauge_y = BOARD_HEIGHT + PIECE_AREA_HEIGHT + SCORE_AREA_HEIGHT - GAUGE_HEIGHT - 5
    # Gauge background.
    pygame.draw.rect(screen, (100, 100, 100), (gauge_x, gauge_y, GAUGE_WIDTH, GAUGE_HEIGHT))
    # Filled portion proportional to speed.
    filled_width = int((speed / 100) * GAUGE_WIDTH)
    pygame.draw.rect(screen, (0, 200, 0), (gauge_x, gauge_y, filled_width, GAUGE_HEIGHT))
    # Gauge border.
    pygame.draw.rect(screen, BOARD_GRID_COLOR, (gauge_x, gauge_y, GAUGE_WIDTH, GAUGE_HEIGHT), 2)
    # Draw a knob at the edge of the filled portion.
    knob_x = gauge_x + filled_width
    knob_y = gauge_y + GAUGE_HEIGHT // 2
    pygame.draw.circle(screen, (255, 255, 255), (knob_x, knob_y), 10)
    pygame.draw.circle(screen, BOARD_GRID_COLOR, (knob_x, knob_y), 10, 2)


def find_auto_move(pieces, board):
    """
    Search for a valid move (left-to-right, top-to-bottom) for any available piece.
    Returns (piece_index, board_x, board_y) if found, or None if no move is possible.
    """
    for piece_index, piece in enumerate(pieces):
        for y in range(BOARD_SIZE):
            for x in range(BOARD_SIZE):
                if can_place_piece(piece, board, x, y):
                    return piece_index, x, y
    return None


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Block Blast - Auto Play with Speed Slider (Dark Mode)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 36)

    board = init_board()
    pieces = generate_pieces()
    score = 0
    game_over = False

    # Speed control parameters.
    max_delay = 800  # Delay (ms) at speed 0.
    min_delay = 200  # Delay (ms) at speed 100.
    # Initial speed (0 to 100) controlled by the slider.
    speed = 50
    slider_dragging = False

    last_move_time = pygame.time.get_ticks()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            # Handle slider dragging.
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = event.pos
                gauge_x = (WINDOW_WIDTH - GAUGE_WIDTH) // 2
                gauge_y = BOARD_HEIGHT + PIECE_AREA_HEIGHT + SCORE_AREA_HEIGHT - GAUGE_HEIGHT - 5
                gauge_rect = pygame.Rect(gauge_x, gauge_y, GAUGE_WIDTH, GAUGE_HEIGHT)
                if gauge_rect.collidepoint(mouse_x, mouse_y):
                    slider_dragging = True
            elif event.type == pygame.MOUSEBUTTONUP:
                slider_dragging = False
            elif event.type == pygame.MOUSEMOTION and slider_dragging:
                mouse_x, _ = event.pos
                gauge_x = (WINDOW_WIDTH - GAUGE_WIDTH) // 2
                # Compute new speed based on mouse X relative to gauge.
                relative_x = mouse_x - gauge_x
                new_speed = int((relative_x / GAUGE_WIDTH) * 100)
                speed = max(0, min(100, new_speed))

        current_time = pygame.time.get_ticks()
        # Calculate current move delay based on the slider-controlled speed.
        current_delay = max_delay - (speed / 100) * (max_delay - min_delay)

        if not game_over and current_time - last_move_time >= current_delay:
            move = find_auto_move(pieces, board)
            if move is not None:
                piece_index, board_x, board_y = move
                place_piece(pieces[piece_index], board, board_x, board_y)
                cleared = check_clear(board)
                score += cleared
                pieces.pop(piece_index)
                if not pieces:
                    pieces = generate_pieces()
            else:
                game_over = True
            last_move_time = current_time

        # Drawing section.
        screen.fill(DARK_BG)
        draw_board(screen, board)
        draw_piece_area(screen, pieces)
        draw_info(screen, score, speed, font)

        if game_over:
            over_text = font.render("Game Over", True, TEXT_COLOR)
            text_x = (WINDOW_WIDTH - over_text.get_width()) // 2
            text_y = (BOARD_HEIGHT - over_text.get_height()) // 2
            screen.blit(over_text, (text_x, text_y))

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    main()
    