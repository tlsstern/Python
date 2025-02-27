import pygame
import random
import heapq

# --- Game Constants ---
CELL_SIZE = 20
BOARD_WIDTH = 20
BOARD_HEIGHT = 20  # number of cells vertically
WINDOW_WIDTH = BOARD_WIDTH * CELL_SIZE
WINDOW_HEIGHT = BOARD_HEIGHT * CELL_SIZE

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)

FRUIT_TYPES = [
    {"name": "apple", "color": (255, 0, 0), "score": 1},
    {"name": "banana", "color": (255, 255, 0), "score": 2},
    {"name": "grape", "color": (128, 0, 128), "score": 3}
]


# --- Helper Functions ---

def heuristic(a, b):
    """Manhattan distance between two cells."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def get_neighbors(node, board_width, board_height):
    """Return neighboring cells (up, down, left, right) within the board."""
    neighbors = []
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
    for dx, dy in directions:
        neighbor = (node[0] + dx, node[1] + dy)
        if 0 <= neighbor[0] < board_width and 0 <= neighbor[1] < board_height:
            neighbors.append(neighbor)
    return neighbors


def a_star(start, goal, obstacles, board_width, board_height):
    """
    A* search algorithm.
    Returns a list of cells representing the path from start (excluded)
    to goal (included) or None if no path is found.
    """
    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), start))
    came_from = {}
    g_score = {start: 0}

    while open_set:
        current = heapq.heappop(open_set)[1]
        if current == goal:
            # Reconstruct the path from goal to start.
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path

        for neighbor in get_neighbors(current, board_width, board_height):
            if neighbor in obstacles:
                continue
            tentative_g = g_score[current] + 1
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f, neighbor))
    return None


def choose_next_move(snake, fruit, board_width, board_height):
    """
    Decide the snake's next move.
    The AI uses A* to find a path from the snake's head to the fruit,
    treating its body (except the head) as obstacles.
    If no path is found, it picks a safe neighboring cell if available.
    """
    head = snake[0]
    obstacles = set(snake[1:])
    path = a_star(head, fruit["pos"], obstacles, board_width, board_height)
    if path is not None and len(path) > 0:
        return path[0]
    else:
        # Fallback: choose any neighboring cell that is not part of the snake.
        safe_moves = [n for n in get_neighbors(head, board_width, board_height)
                      if n not in snake]
        if safe_moves:
            return random.choice(safe_moves)
        else:
            possible_moves = get_neighbors(head, board_width, board_height)
            if possible_moves:
                return random.choice(possible_moves)
            else:
                return None


def place_fruit(snake, board_width, board_height):
    """
    Place a fruit at a random cell that is not occupied by the snake.
    Randomly selects a fruit type.
    """
    occupied = set(snake)
    while True:
        pos = (random.randint(0, board_width - 1), random.randint(0, board_height - 1))
        if pos not in occupied:
            fruit_type = random.choice(FRUIT_TYPES)
            return {"pos": pos, "type": fruit_type}


# --- Main Game Loop ---

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Self-Playing Snake")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 36)

    score = 0
    speed = 15  # frames per second.

    # Initialize the snake with 3 segments.
    snake = [
        (BOARD_WIDTH // 2, BOARD_HEIGHT // 2),
        (BOARD_WIDTH // 2 - 1, BOARD_HEIGHT // 2),
        (BOARD_WIDTH // 2 - 2, BOARD_HEIGHT // 2)
    ]
    fruit = place_fruit(snake, BOARD_WIDTH, BOARD_HEIGHT)

    running = True
    while running:
        clock.tick(speed)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Decide the next move using the AI.
        next_move = choose_next_move(snake, fruit, BOARD_WIDTH, BOARD_HEIGHT)
        if next_move is None:
            print("No safe moves available. Game over.")
            running = False
            continue

        # Move the snake by inserting the new head.
        snake.insert(0, next_move)

        # Check if the snake has eaten the fruit.
        if next_move == fruit["pos"]:
            score += fruit["type"]["score"]
            fruit = place_fruit(snake, BOARD_WIDTH, BOARD_HEIGHT)
        else:
            # Remove the tail if no fruit is eaten.
            snake.pop()

        # Check for self-collision.
        if snake[0] in snake[1:]:
            print("Snake collided with itself. Game over.")
            running = False
            continue

        # Draw everything.
        screen.fill(BLACK)

        # Draw the snake.
        for segment in snake:
            rect = pygame.Rect(segment[0] * CELL_SIZE,
                               segment[1] * CELL_SIZE,
                               CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, GREEN, rect)

        # Draw the fruit.
        fruit_rect = pygame.Rect(fruit["pos"][0] * CELL_SIZE,
                                 fruit["pos"][1] * CELL_SIZE,
                                 CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, fruit["type"]["color"], fruit_rect)

        # Render and display the score.
        score_surface = font.render("Score: " + str(score), True, WHITE)
        screen.blit(score_surface, (10, 10))

        pygame.display.flip()

    print("Final Score:", score)
    pygame.quit()


if __name__ == "__main__":
    main()
