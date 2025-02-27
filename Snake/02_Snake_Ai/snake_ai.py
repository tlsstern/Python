import pygame
import random
import sys
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# --------------------------
# Snake Game Environment
# --------------------------
class SnakeGameAI:
    def __init__(self, w=400, h=400):
        self.w = w  # width of the game display
        self.h = h  # height of the game display
        self.block_size = 20  # each block is 20x20; grid is 20x20
        self.reset()

        pygame.init()
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Snake AI")
        self.clock = pygame.time.Clock()

    def reset(self):
        self.direction = (1, 0)  # initial direction (right)
        self.head = [self.w // 2, self.h // 2]
        self.snake = [self.head.copy()]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w - self.block_size) // self.block_size) * self.block_size
        y = random.randint(0, (self.h - self.block_size) // self.block_size) * self.block_size
        self.food = [x, y]
        if self.food in self.snake:
            self._place_food()

    def _manhattan_distance(self, pt1, pt2):
        return abs(pt1[0] - pt2[0]) + abs(pt1[1] - pt2[1])

    def play_step(self, action):
        """
        Executes one time step.
          action: one-hot encoded list with 3 elements [straight, right turn, left turn]
        Returns: reward, game_over (bool), score
        """
        self.frame_iteration += 1

        # Process pygame events (e.g., window close)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Calculate distance to food before moving
        old_distance = self._manhattan_distance(self.head, self.food)

        # Move the snake
        self._move(action)
        self.snake.insert(0, self.head.copy())

        reward = 0
        game_over = False

        # Check for collisions or if the snake has taken too long
        if self._is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            # Final reward upon death is based on fruits eaten minus a death penalty of 50.
            reward = (self.score * 25) - 75
            return reward, game_over, self.score

        # Check if food is eaten
        if self.head == self.food:
            self.score += 1
            reward = 20  # fruit reward changed from 50 to 10
            self._place_food()
        else:
            self.snake.pop()

        # Reward shaping: reward change due to moving closer or further away from the food
        new_distance = self._manhattan_distance(self.head, self.food)
        distance_reward = (old_distance - new_distance) * 0.1  # positive if getting closer
        reward += distance_reward

        # Time penalty to encourage faster solutions
        # reward -= 0.1

        self._update_ui()
        # Increase game speed by setting a higher FPS (adjust as needed)
        self.clock.tick(2000)
        return reward, game_over, self.score

    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # Check wall collisions
        if pt[0] < 0 or pt[0] > self.w - self.block_size or pt[1] < 0 or pt[1] > self.h - self.block_size:
            return True
        # Check self-collision
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        self.display.fill((0, 0, 0))
        # Draw snake (green)
        for pt in self.snake:
            pygame.draw.rect(self.display, (0, 255, 0),
                             pygame.Rect(pt[0], pt[1], self.block_size, self.block_size))
        # Draw food (red)
        pygame.draw.rect(self.display, (255, 0, 0),
                         pygame.Rect(self.food[0], self.food[1], self.block_size, self.block_size))
        pygame.display.flip()

    def _move(self, action):
        """
        Updates snake's direction and head based on the action.
        Action is one-hot encoded:
          [1, 0, 0] -> straight
          [0, 1, 0] -> right turn
          [0, 0, 1] -> left turn
        """
        clock_wise = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        idx = clock_wise.index(tuple(self.direction))

        if np.array_equal(action, [1, 0, 0]):  # no change
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):  # right turn
            new_dir = clock_wise[(idx + 1) % 4]
        else:  # left turn
            new_dir = clock_wise[(idx - 1) % 4]

        self.direction = new_dir
        self.head[0] += self.direction[0] * self.block_size
        self.head[1] += self.direction[1] * self.block_size

    def get_state(self):
        """
        Returns a 13-dimensional state:
          0-2: Danger straight, right, left (booleans)
          3-6: Current direction (left, right, up, down) as booleans
          7-10: Food location relative to head (booleans: food left/right/up/down)
          11-12: Normalized differences between food and head (continuous values)
        """
        head = self.head.copy()
        point_l = [head[0] - self.block_size, head[1]]
        point_r = [head[0] + self.block_size, head[1]]
        point_u = [head[0], head[1] - self.block_size]
        point_d = [head[0], head[1] + self.block_size]

        dir_l = self.direction == (-1, 0)
        dir_r = self.direction == (1, 0)
        dir_u = self.direction == (0, -1)
        dir_d = self.direction == (0, 1)

        state = [
            # Danger straight
            (dir_r and self._is_collision(point_r)) or
            (dir_l and self._is_collision(point_l)) or
            (dir_u and self._is_collision(point_u)) or
            (dir_d and self._is_collision(point_d)),

            # Danger right
            (dir_u and self._is_collision(point_r)) or
            (dir_d and self._is_collision(point_l)) or
            (dir_l and self._is_collision(point_u)) or
            (dir_r and self._is_collision(point_d)),

            # Danger left
            (dir_d and self._is_collision(point_r)) or
            (dir_u and self._is_collision(point_l)) or
            (dir_r and self._is_collision(point_u)) or
            (dir_l and self._is_collision(point_d)),

            # Current direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location (booleans)
            self.food[0] < head[0],  # food left of head
            self.food[0] > head[0],  # food right of head
            self.food[1] < head[1],  # food above head
            self.food[1] > head[1],  # food below head
        ]

        # Add continuous features: normalized differences between food and head
        state.append((self.food[0] - head[0]) / self.w)
        state.append((self.food[1] - head[1]) / self.h)

        return np.array(state, dtype=int)


# --------------------------
# Deep Q-Network (DQN) Model
# --------------------------
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Linear_QNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x


# --------------------------
# Q-Learning Trainer
# --------------------------
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.lr = lr
        self.gamma = gamma
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        pred = self.model(state)
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()


# --------------------------
# DQN Agent
# --------------------------
class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0  # exploration vs. exploitation parameter
        self.gamma = 0.9  # discount factor
        self.memory = deque(maxlen=100_000)  # experience replay memory
        self.model = Linear_QNet(13, 256, 3)
        self.trainer = QTrainer(self.model, lr=0.005, gamma=self.gamma)

    def get_state(self, game):
        return game.get_state()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > 1000:
            mini_sample = random.sample(self.memory, 1000)
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games  # exploration rate decays with number of games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move


# --------------------------
# Saving and Loading Model Functions
# --------------------------
def save_model(agent, record, filename='model.pth'):
    print("Saving model with record:", record)
    torch.save({
         'n_games': agent.n_games,
         'record': record,
         'model_state_dict': agent.model.state_dict(),
         'optimizer_state_dict': agent.trainer.optimizer.state_dict(),
    }, filename)

def load_model(filename='model.pth'):
    checkpoint = torch.load(filename)
    agent = Agent()
    agent.model.load_state_dict(checkpoint['model_state_dict'])
    agent.trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    agent.n_games = checkpoint['n_games']
    record = checkpoint['record']
    print("Loaded model with record:", record)
    return agent, record


# --------------------------
# Training Loop
# --------------------------
def train():
    # Attempt to load an existing model; otherwise, start fresh.
    try:
        agent, record = load_model('model.pth')
    except FileNotFoundError:
        agent = Agent()
        record = 0

    game = SnakeGameAI()
    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                save_model(agent, record)

            print('Game', agent.n_games, 'Score', score, 'Record:', record, "Reward:", reward)


if __name__ == '__main__':
    train()
