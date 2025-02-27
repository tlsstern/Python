import multiprocessing as mp
import random
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
from collections import deque

# (Include the definitions for SnakeGameAI, Linear_QNet, QTrainer, and Agent here.)
# For brevity, we assume you have the complete definitions from the previous code,
# with an added "render" parameter to optionally disable display updates.

class SnakeGameAI:
    def __init__(self, w=400, h=400, render=True):
        self.w = w
        self.h = h
        self.block_size = 20
        self.render = render
        self.reset()
        if self.render:
            pygame.init()
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption("Snake AI")
            self.clock = pygame.time.Clock()

    def reset(self):
        self.direction = (1, 0)
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
        self.frame_iteration += 1
        # In headless mode, skip event processing and UI updates.
        if self.render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

        old_distance = self._manhattan_distance(self.head, self.food)
        self._move(action)
        self.snake.insert(0, self.head.copy())
        reward = 0
        game_over = False

        if self._is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            if self.head in self.snake[1:]:
                reward = -20
            else:
                reward = -10
            return reward, game_over, self.score

        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        new_distance = self._manhattan_distance(self.head, self.food)
        if new_distance < old_distance:
            reward += 0.2
        else:
            reward -= 0.2

        if self.render:
            self._update_ui()
            self.clock.tick(120)
        return reward, game_over, self.score

    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt[0] < 0 or pt[0] > self.w - self.block_size or pt[1] < 0 or pt[1] > self.h - self.block_size:
            return True
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        self.display.fill((0, 0, 0))
        for pt in self.snake:
            pygame.draw.rect(self.display, (0, 255, 0),
                             pygame.Rect(pt[0], pt[1], self.block_size, self.block_size))
        pygame.draw.rect(self.display, (255, 0, 0),
                         pygame.Rect(self.food[0], self.food[1], self.block_size, self.block_size))
        pygame.display.flip()

    def _move(self, action):
        clock_wise = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        idx = clock_wise.index(tuple(self.direction))
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = clock_wise[(idx + 1) % 4]
        else:
            new_dir = clock_wise[(idx - 1) % 4]
        self.direction = new_dir
        self.head[0] += self.direction[0] * self.block_size
        self.head[1] += self.direction[1] * self.block_size

    def get_state(self):
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
            (dir_r and self._is_collision(point_r)) or
            (dir_l and self._is_collision(point_l)) or
            (dir_u and self._is_collision(point_u)) or
            (dir_d and self._is_collision(point_d)),
            (dir_u and self._is_collision(point_r)) or
            (dir_d and self._is_collision(point_l)) or
            (dir_l and self._is_collision(point_u)) or
            (dir_r and self._is_collision(point_d)),
            (dir_d and self._is_collision(point_r)) or
            (dir_u and self._is_collision(point_l)) or
            (dir_r and self._is_collision(point_u)) or
            (dir_l and self._is_collision(point_d)),
            dir_l, dir_r, dir_u, dir_d,
            self.food[0] < head[0],
            self.food[0] > head[0],
            self.food[1] < head[1],
            self.food[1] > head[1],
            (self.food[0] - head[0]) / self.w,
            (self.food[1] - head[1]) / self.h,
        ]
        return np.array(state, dtype=int)

# (Linear_QNet, QTrainer, and Agent definitions remain the same as before.)

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Linear_QNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
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
            done = (done, )
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

class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=100_000)
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
        self.epsilon = 80 - self.n_games
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

# Example worker function that runs an episode without rendering.
def run_episode(_):
    # Run in headless mode for speed
    game = SnakeGameAI(render=False)
    agent = Agent()
    while True:
        state_old = agent.get_state(game)
        final_move = agent.get_action(state_old)
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)
        agent.train_short_memory(state_old, final_move, reward, state_new, done)
        agent.remember(state_old, final_move, reward, state_new, done)
        if done:
            break
    return score

if __name__ == '__main__':
    # Using a pool to run multiple episodes in parallel.
    num_processes = 8  # Number of parallel processes
    num_episodes = 100  # Total episodes to run
    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(run_episode, range(num_episodes))
    print("Episode scores:", results)
