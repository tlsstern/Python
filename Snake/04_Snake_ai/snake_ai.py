import pygame
import random
import numpy as np
import collections

# --- Game Constants ---
CELL_SIZE = 20
BOARD_WIDTH = 20
BOARD_HEIGHT = 20
WINDOW_WIDTH = BOARD_WIDTH * CELL_SIZE
WINDOW_HEIGHT = BOARD_HEIGHT * CELL_SIZE

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)

FRUIT_TYPES = [
    {"name": "apple", "color": RED, "score": 1},
    {"name": "banana", "color": YELLOW, "score": 5},  # Banana more valuable
    {"name": "grape", "color": PURPLE, "score": 10}  # Grape most valuable
]

# --- Action Definitions ---
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

# --- Q-Learning Parameters ---
LEARNING_RATE = 0.001
DISCOUNT_FACTOR = 0.9
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.0001
MAX_MEMORY_LENGTH = 50000
BATCH_SIZE = 64

# --- Training Parameters ---
N_EPISODES = 500000
MAX_STEPS_PER_EPISODE = 500
SHOW_EVERY = N_EPISODES

# --- Rewards ---
#  Make fruit rewards significantly larger, and scale them with fruit value.
REWARD_FRUIT_BASE = 100
REWARD_DEATH = -500
REWARD_STEP = -0.01
REWARD_CLOSER = 1 / (BOARD_WIDTH + BOARD_HEIGHT)
EARLY_DEATH_PENALTY = -10

# --- Discretization Parameters ---
X_BINS = 5
Y_BINS = 5
USE_DANGER = True

# --- Helper functions ---

def place_fruit(snake, board_width, board_height):
    occupied = set(snake)
    while True:
        pos = (random.randint(0, board_width - 1), random.randint(0, board_height - 1))
        if pos not in occupied:
            fruit_type = random.choice(FRUIT_TYPES)
            return {"pos": pos, "type": fruit_type}

def get_state(snake, fruit, board_width, board_height):
    head = snake[0]
    fruit_x_rel = fruit["pos"][0] - head[0]
    fruit_y_rel = fruit["pos"][1] - head[1]

    if len(snake) > 1:
        dx = head[0] - snake[1][0]
        dy = head[1] - snake[1][1]
    else:
        dx, dy = 0, -1

    if (dx, dy) == (0, -1):
        point_ahead = (head[0], head[1] - 1)
    elif (dx, dy) == (0, 1):
        point_ahead = (head[0], head[1] + 1)
    elif (dx, dy) == (-1, 0):
        point_ahead = (head[0] - 1, head[1])
    else:
        point_ahead = (head[0] + 1, head[1])

    danger_ahead = (point_ahead in snake or
                    point_ahead[0] < 0 or point_ahead[0] >= board_width or
                    point_ahead[1] < 0 or point_ahead[1] >= board_height)

    point_u = (head[0], head[1] - 1)
    point_d = (head[0], head[1] + 1)
    point_l = (head[0] - 1, head[1])
    point_r = (head[0] + 1, head[1])

    danger_up = point_u in snake or point_u[1] < 0
    danger_down = point_d in snake or point_d[1] >= board_height
    danger_left = point_l in snake or point_l[0] < 0
    danger_right = point_r in snake or point_r[0] >= board_width


    if USE_DANGER:
      state = [fruit_x_rel, fruit_y_rel, danger_up, danger_down, danger_left, danger_right]
    else:
      state = [fruit_x_rel, fruit_y_rel]
    return np.array(state, dtype=int)

class SnakeGameAI:
    def __init__(self, board_width, board_height):
        self.board_width = board_width
        self.board_height = board_height
        self.state_size = X_BINS * Y_BINS * (2**4 if USE_DANGER else 1)
        self.action_size = 4
        self.q_table = np.random.uniform(low=-0.1, high=0.1, size=(self.state_size, self.action_size))
        self.epsilon = EPSILON_START
        self.memory = collections.deque(maxlen=MAX_MEMORY_LENGTH)
        self.episode_count = 0
        self.reset()

    def reset(self):
        self.snake = [
            (self.board_width // 2, self.board_height // 2),
            (self.board_width // 2 - 1, self.board_height // 2),
            (self.board_width // 2 - 2, self.board_height // 2)
        ]
        self.fruit = place_fruit(self.snake, self.board_width, self.board_height)
        self.score = 0
        self.steps = 0
        self.previous_snake_length = len(self.snake) # For early death detection.
        return get_state(self.snake, self.fruit, self.board_width, self.board_height)

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            return np.argmax(self.q_table[self.state_to_int(state)])

    def step(self, action):
        self.steps += 1
        prev_state_full = get_state(self.snake, self.fruit, self.board_width, self.board_height)
        prev_head = self.snake[0]

        if action == UP:
            new_head = (self.snake[0][0], self.snake[0][1] - 1)
        elif action == DOWN:
            new_head = (self.snake[0][0], self.snake[0][1] + 1)
        elif action == LEFT:
            new_head = (self.snake[0][0] - 1, self.snake[0][1])
        elif action == RIGHT:
            new_head = (self.snake[0][0] + 1, self.snake[0][1])
        else:
            raise ValueError("Invalid action")

        prev_distance = abs(prev_head[0] - self.fruit["pos"][0]) + abs(prev_head[1] - self.fruit["pos"][1])
        self.snake.insert(0, new_head)
        new_distance = abs(new_head[0] - self.fruit["pos"][0]) + abs(new_head[1] - self.fruit["pos"][1])

        reward = REWARD_STEP

        if new_head == self.fruit["pos"]:
            self.score += self.fruit["type"]["score"]
            # Scale fruit reward by fruit score
            reward += REWARD_FRUIT_BASE * self.fruit["type"]["score"]
            self.fruit = place_fruit(self.snake, self.board_width, self.board_height)
            self.previous_snake_length = len(self.snake) # Reset length after eating.
        else:
            self.snake.pop()
            reward += (prev_distance - new_distance) * REWARD_CLOSER

        game_over = False
        if (new_head[0] < 0 or new_head[0] >= self.board_width or
                new_head[1] < 0 or new_head[1] >= self.board_height or
                new_head in self.snake[1:]):
            game_over = True
            reward = REWARD_DEATH

            # --- Early Death Penalty ---
            if len(self.snake) < self.previous_snake_length:  # Died before growing
                reward += EARLY_DEATH_PENALTY
            self.previous_snake_length = len(self.snake)

        next_state = get_state(self.snake, self.fruit, self.board_width, self.board_height)
        self.remember(prev_state_full, action, reward, next_state, game_over)
        self.replay()
        return next_state, reward, game_over, prev_state_full

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < BATCH_SIZE:
            return

        minibatch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + DISCOUNT_FACTOR * np.max(self.q_table[self.state_to_int(next_state)])
            self.q_table[self.state_to_int(state), action] = (1 - LEARNING_RATE) * self.q_table[self.state_to_int(state), action] + LEARNING_RATE * target

    def state_to_int(self, state):
        """Converts the discretized state to an integer for Q-table indexing."""
        if USE_DANGER:
            fruit_x_rel, fruit_y_rel, danger_up, danger_down, danger_left, danger_right = state
            # Discretize x and y coordinates
            x_bin = self.discretize_value(fruit_x_rel, self.board_width, X_BINS)
            y_bin = self.discretize_value(fruit_y_rel, self.board_height, Y_BINS)
            # Convert danger booleans to integer
            danger_int = int(f"{int(danger_up)}{int(danger_down)}{int(danger_left)}{int(danger_right)}", 2)
            # Combine bin indices and danger into single integer
            state_int = x_bin + y_bin * X_BINS + danger_int * X_BINS * Y_BINS
        else:
            fruit_x_rel, fruit_y_rel = state
            x_bin = self.discretize_value(fruit_x_rel, self.board_width, X_BINS)
            y_bin = self.discretize_value(fruit_y_rel, self.board_height, Y_BINS)
            state_int = x_bin + y_bin * X_BINS
        return state_int

    def discretize_value(self, value, max_value, num_bins):
        """Discretizes a continuous value into a bin index."""
        bin_width = (2 * max_value) / num_bins
        bin_index = int((value + max_value) // bin_width)
        bin_index = max(0, min(bin_index, num_bins - 1))
        return bin_index

    def decay_epsilon(self):
        self.epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * np.exp(-EPSILON_DECAY * self.episode_count)

# --- Main Game Loop ---

def main():
    game = SnakeGameAI(BOARD_WIDTH, BOARD_HEIGHT)
    scores = []

    for episode in range(N_EPISODES):
        state = game.reset()
        done = False
        total_reward = 0

        for step in range(MAX_STEPS_PER_EPISODE):
            action = game.get_action(state)
            next_state, reward, done, prev_state = game.step(action)
            total_reward += reward
            state = next_state

            if done:
                break

        game.episode_count += 1
        game.decay_epsilon()
        scores.append(game.score)
        print(f"Episode: {episode + 1}, Score: {game.score}, Epsilon: {game.epsilon:.4f}, Total Steps: {step}, Total Reward: {total_reward}")

    np.save('snake_q_table.npy', game.q_table)
    print("Training Complete!")
    print("Average Score:", sum(scores) / len(scores))

    # --- Run a final episode with visualization ---
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Self-Playing Snake - Final Run")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 36)

    game.q_table = np.load('snake_q_table.npy')
    game.epsilon = 0
    state = game.reset()
    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        action = game.get_action(state)
        next_state, reward, done, _ = game.step(action)
        state = next_state

        screen.fill(BLACK)
        for segment in game.snake:
            rect = pygame.Rect(segment[0] * CELL_SIZE, segment[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, GREEN, rect)

        fruit_rect = pygame.Rect(game.fruit["pos"][0] * CELL_SIZE, game.fruit["pos"][1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, game.fruit["type"]["color"], fruit_rect)
        score_surface = font.render(f"Score: {game.score}", True, WHITE)
        screen.blit(score_surface, (10, 10))

        pygame.display.flip()
        clock.tick(15)

    pygame.quit()

if __name__ == "__main__":
    main()