import pygame, sys, os, gym
import numpy as np
import tensorflow as tf
from collections import deque
from rl.agents import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from random import choice, randint

from player import Player
import obstacle
from alien import Alien, Extra
from laser import Laser

# Enable eager execution
tf.config.run_functions_eagerly(True)
 
class Game(gym.Env):
	def __init__(self):
		'''initialization'''

		# Player setup
		player_sprite = Player((screen_width / 2, screen_height), screen_width, 5)
		self.player = pygame.sprite.GroupSingle(player_sprite)

		# health and score setup
		self.lives = 3
		self.live_surf = pygame.image.load(os.path.join('graphics', 'player.png'))
		self.live_x_start_pos = screen_width - (self.live_surf.get_size()[0] * 2 + 20)
		self.score = 0
		self.font = pygame.font.Font(os.path.join('font', 'Pixeled.ttf'), 20)

		# Obstacle setup
		self.shape = obstacle.shape
		self.block_size = 6
		self.blocks = pygame.sprite.Group()
		self.obstacle_amount = 4
		self.obstacle_x_positions = [num * (screen_width / self.obstacle_amount) for num in range(self.obstacle_amount)]
		self.create_multiple_obstacles(*self.obstacle_x_positions, x_start = screen_width / 15, y_start = 480)

		# Alien setup
		self.aliens = pygame.sprite.Group()
		self.alien_lasers = pygame.sprite.Group()
		self.alien_setup(rows = 6, cols = 8)
		self.alien_direction = 1
		self.alien_value = 10

		# Extra setup
		self.extra = pygame.sprite.GroupSingle()
		self.extra_spawn_time = randint(40, 80)

		# Audio
		music = pygame.mixer.Sound(os.path.join('audio', 'music.wav'))
		music.set_volume(0.2)
		music.play(loops = -1)
		self.laser_sound = pygame.mixer.Sound(os.path.join('audio', 'laser.wav'))
		self.laser_sound.set_volume(0.5)
		self.explosion_sound = pygame.mixer.Sound(os.path.join('audio', 'explosion.wav'))
		self.explosion_sound.set_volume(0.3)

		# Set up the observation space
		self.observation_space = gym.spaces.Box(low=0, high=255, shape=(screen_height, screen_width, 3), dtype=np.uint8)

		# Set up the action space
		self.action_space = gym.spaces.Discrete(3)
		
		# Build the deep Q-network model for the agent
		self.model = self.build_model(screen_height, screen_width, 3, self.action_space.n)
		
		# Exploration parameters
		self.epsilon = 1.0  # Initial exploration rate
		self.epsilon_min = 0.01  # Minimum exploration rate
		self.epsilon_decay = 0.995  # Decay rate for exploration rate
		
		# Memory buffer for experience replay
		self.memory = deque(maxlen=2000)  # Replay memory buffer size
		
		# Hyperparameters for training
		self.batch_size = 32  # Batch size for training the neural network
		self.gamma = 0.95  # Discount factor for future rewards

		# Build the DQN agent
		self.dqn_agent = self.build_agent(self.model, self.action_space.n)
		
	def build_model(self, height, width, channels, actions):
		model = tf.keras.Sequential()
		model.add(tf.keras.layers.Convolution2D(32, (8,8), strides=(4,4), activation='relu', input_shape=(height, width, channels)))
		model.add(tf.keras.layers.Convolution2D(64, (4,4), strides=(2,2), activation='relu'))
		model.add(tf.keras.layers.Convolution2D(64, (3,3), activation='relu'))
		model.add(tf.keras.layers.Flatten())
		model.add(tf.keras.layers.Dense(512, activation='relu'))
		model.add(tf.keras.layers.Dense(256, activation='relu'))
		model.add(tf.keras.layers.Dense(actions, activation='linear'))
		return model
	
	def build_agent(self, model, actions):
		policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1, value_min=.1, value_test=.2, nb_steps=10000)
		memory = SequentialMemory(limit=1000, window_length=3)
		dqn = DQNAgent(model=model, memory=memory, policy=policy, enable_dueling_network=True,
                       dueling_type='avg', nb_actions=actions, nb_steps_warmup=1000)
		return dqn
	
	def reset(self):
		'''Reset the environment and return the initial observation'''

		# Player setup
		player_sprite = Player((screen_width / 2, screen_height), screen_width, 5)
		self.player = pygame.sprite.GroupSingle(player_sprite)

		# health and score setup
		self.lives = 3
		self.live_surf = pygame.image.load(os.path.join('graphics', 'player.png'))
		self.live_x_start_pos = screen_width - (self.live_surf.get_size()[0] * 2 + 20)
		self.score = 0
		self.font = pygame.font.Font(os.path.join('font', 'Pixeled.ttf'), 20)

		# Obstacle setup
		self.shape = obstacle.shape
		self.block_size = 6
		self.blocks = pygame.sprite.Group()
		self.obstacle_amount = 4
		self.obstacle_x_positions = [num * (screen_width / self.obstacle_amount) for num in range(self.obstacle_amount)]
		self.create_multiple_obstacles(*self.obstacle_x_positions, x_start = screen_width / 15, y_start = 480)

		# Alien setup
		self.aliens = pygame.sprite.Group()
		self.alien_lasers = pygame.sprite.Group()
		self.alien_setup(rows = 6, cols = 8)
		self.alien_direction = 1

		# Extra setup
		self.extra = pygame.sprite.GroupSingle()
		self.extra_spawn_time = randint(40, 80)

		# Audio
		music = pygame.mixer.Sound(os.path.join('audio', 'music.wav'))
		music.set_volume(0.2)
		music.play(loops = -1)
		self.laser_sound = pygame.mixer.Sound(os.path.join('audio', 'laser.wav'))
		self.laser_sound.set_volume(0.5)
		self.explosion_sound = pygame.mixer.Sound(os.path.join('audio', 'explosion.wav'))
		self.explosion_sound.set_volume(0.3)

		observation = self.get_observation()
		
		return observation
	
	def step(self, action):
		# Validate the action
		if not self.action_space.contains(action):
			raise ValueError(f"Invalid action: {action}")
		
		# Take the action and update the game state
		if action == 0:  # Move left
			self.player.sprite.move_left()
		elif action == 1:  # Move right
			self.player.sprite.move_right()
		elif action == 2:  # Fire player laser
			self.player.sprite.fire_laser()

		# Move aliens horizontally based on the alien_direction variable
		alien_speed = 2
		for alien in self.aliens:
			alien.rect.x += alien_speed * self.alien_direction

		# Move aliens downward when they reach the screen edge
		if self.aliens:
			if self.aliens.sprites()[0].rect.right >= screen_width or self.aliens.sprites()[-1].rect.left <= 0:
				self.alien_direction *= -1
				self.alien_move_down(2)
				
		# Check for collisions with player lasers
		if self.player.sprite.lasers:
			for laser in self.player.sprite.lasers:
				# Check for collisions with obstacles
				if pygame.sprite.spritecollide(laser, self.blocks, True):
					laser.kill()
					
				# Check for collisions with aliens
				aliens_hit = pygame.sprite.spritecollide(laser, self.aliens, True)
				if aliens_hit:
					for alien in aliens_hit:
						self.score += alien.value
					laser.kill()
					self.explosion_sound.play()
				
				# Check for collisions with extra aliens
				if pygame.sprite.spritecollide(laser, self.extra, True):
					self.score += 500
					laser.kill()			
					
		# Move alien lasers downward
		alien_laser_speed = 4
		for alien_laser in self.alien_lasers:
			alien_laser.rect.y += alien_laser_speed
			
		# Check for collisions with player
		if self.alien_lasers:
			for alien_laser in self.alien_lasers:
				if pygame.sprite.spritecollide(alien_laser, self.blocks, True):
					alien_laser.kill()
					
				if pygame.sprite.spritecollide(alien_laser, self.player, False):
					alien_laser.kill()
					self.lives -= 1
					if self.lives <= 0:
						pygame.quit()
						sys.exit()
						
		# Check for collisions with aliens and obstacles
		for alien in self.aliens:
			pygame.sprite.spritecollide(alien, self.blocks, True)
			
			if pygame.sprite.spritecollide(alien, self.player, False):
				pygame.quit()
				sys.exit()


		# Get the new observation after taking the action
		observation = self.get_observation()

		reward = 0

		# Positive reward for destroying an alien
		destroyed_aliens = len(self.aliens) - len(pygame.sprite.spritecollide(self.aliens, self.blocks, False))
		reward += destroyed_aliens * self.alien_value

		# Negative reward for getting hit by an alien laser or colliding with an obstacle
		if pygame.sprite.spritecollide(self.player.sprite, self.alien_lasers, False) or pygame.sprite.spritecollide(self.player.sprite, self.blocks, False):
			reward -= 10
		
		# Check if the game is over
		done = False
		if self.lives <= 0:
			done = True
	    
		# return the player's current score, lives
		info = {'score': self.score, 'lives': self.lives}
		
		return observation, reward, done, info
	
	def render(self, mode='human'):
		# Implement the rendering method (optional)
		pass

	def close(self):
		# Implement the close method (optional)
		pass
	
	def create_obstacle(self, x_start, y_start, offset_x):
		for row_index, row in enumerate(self.shape):
			for col_index, col in enumerate(row):
				if col == 'x':
					x = x_start + col_index * self.block_size + offset_x
					y = y_start + row_index * self.block_size
					block = obstacle.Block(self.block_size, (241, 79, 80), x, y)
					self.blocks.add(block)

	def create_multiple_obstacles(self, *offset, x_start, y_start):
		for offset_x in offset:
			self.create_obstacle(x_start, y_start, offset_x)

	def alien_setup(self, rows, cols, x_distance = 60, y_distance = 48, x_offset = 70, y_offset = 100):
		for row_index, row in enumerate(range(rows)):
			for col_index, col in enumerate(range(cols)):
				x = col_index * x_distance + x_offset
				y = row_index * y_distance + y_offset
				
				if row_index == 0: alien_sprite = Alien('yellow', x, y)
				elif 1 <= row_index <= 2: alien_sprite = Alien('green', x, y)
				else: alien_sprite = Alien('red', x, y)
				self.aliens.add(alien_sprite)

	def alien_position_checker(self):
		all_aliens = self.aliens.sprites()
		for alien in all_aliens:
			if alien.rect.right >= screen_width:
				self.alien_direction = -1
				self.alien_move_down(2)
			elif alien.rect.left <= 0:
				self.alien_direction = 1
				self.alien_move_down(2)

	def alien_move_down(self, distance):
		if self.aliens:
			for alien in self.aliens.sprites():
				alien.rect.y += distance

	def alien_shoot(self):
		if self.aliens.sprites():
			random_alien = choice(self.aliens.sprites())
			laser_sprite = Laser(random_alien.rect.center, 6, screen_height)
			self.alien_lasers.add(laser_sprite)
			self.laser_sound.play()

	def extra_alien_timer(self):
		self.extra_spawn_time -= 1
		if self.extra_spawn_time <= 0:
			self.extra.add(Extra(choice(['right', 'left']), screen_width))
			self.extra_spawn_time = randint(400,800)

	def collision_checks(self):

		# player lasers 
		if self.player.sprite.lasers:
			for laser in self.player.sprite.lasers:
				# obstacle collisions
				if pygame.sprite.spritecollide(laser, self.blocks, True):
					laser.kill()
					

				# alien collisions
				aliens_hit = pygame.sprite.spritecollide(laser, self.aliens, True)
				if aliens_hit:
					for alien in aliens_hit:
						self.score += alien.value
					laser.kill()
					self.explosion_sound.play()

				# extra collision
				if pygame.sprite.spritecollide(laser, self.extra, True):
					self.score += 500
					laser.kill()

		# alien lasers 
		if self.alien_lasers:
			for laser in self.alien_lasers:
				# obstacle collisions
				if pygame.sprite.spritecollide(laser, self.blocks, True):
					laser.kill()

				if pygame.sprite.spritecollide(laser, self.player, False):
					laser.kill()
					self.lives -= 1
					if self.lives <= 0:
						pygame.quit()
						sys.exit()

		# aliens
		if self.aliens:
			for alien in self.aliens:
				pygame.sprite.spritecollide(alien, self.blocks, True)

				if pygame.sprite.spritecollide(alien, self.player, False):
					pygame.quit()
					sys.exit()
	
	def display_lives(self):
		for live in range(self.lives - 1):
			x = self.live_x_start_pos + (live * (self.live_surf.get_size()[0] + 10))
			screen.blit(self.live_surf , (x, 8))

	def display_score(self):
		score_surf = self.font.render(f'score: {self.score}', False, 'white')
		score_rect = score_surf.get_rect(topleft = (10, -10))
		screen.blit(score_surf, score_rect)

	def victory_message(self):
		if not self.aliens.sprites():
			victory_surf = self.font.render('You won', False, 'white')
			victory_rect = victory_surf.get_rect(center = (screen_width / 2, screen_height / 2))
			screen.blit(victory_surf, victory_rect)

	def get_observation(self):

		# This function captures the current game screen and player's position
		player_pos = self.player.sprite.rect.center

		# Get the current game screen
		game_screen = pygame.surfarray.array3d(screen)
		game_screen = np.swapaxes(game_screen, 0, 1)  # Swap axes to match Gym convention (height, width, channels)

		# Normalize the values to 0-255 and convert to uint8
		game_screen = np.clip((game_screen / 255.0) * 255.0, 0, 255).astype(np.uint8)

		# Add player's position as an extra feature in the observation
		player_position_indicator = np.zeros_like(game_screen)
		player_position_indicator[player_pos[1], player_pos[0]] = [255, 255, 255]

		# Combine the game screen and player position indicator to form the final observation
		observation = np.maximum(game_screen, player_position_indicator)
        
		return observation

	def run(self):
		self.player.update()
		self.alien_lasers.update()
		self.extra.update()
		
		self.aliens.update(self.alien_direction)
		self.alien_position_checker()
		self.extra_alien_timer()
		self.collision_checks()
		
		self.player.sprite.lasers.draw(screen)
		self.player.draw(screen)
		self.blocks.draw(screen)
		self.aliens.draw(screen)
		self.alien_lasers.draw(screen)
		self.extra.draw(screen)
		self.display_lives()
		self.display_score()
		self.victory_message()

class CRT:
	def __init__(self):
		self.tv = pygame.image.load(os.path.join('graphics', 'tv.png'))
		self.tv = pygame.transform.scale(self.tv, (screen_width, screen_height))

	def create_crt_lines(self):
		line_height = 3
		line_amount = int(screen_height / line_height)
		for line in range(line_amount):
			y_pos = line * line_height
			pygame.draw.line(self.tv, 'black', (0, y_pos), (screen_width, y_pos), 1)

	def draw(self):
		self.tv.set_alpha(randint(75, 90))
		self.create_crt_lines()
		screen.blit(self.tv, (0, 0))

if __name__ == '__main__':
	pygame.init()
	screen_width = 600
	screen_height = 600
	screen = pygame.display.set_mode((screen_width, screen_height))
	clock = pygame.time.Clock()
	env = Game()
	crt = CRT()

	ALIENLASER = pygame.USEREVENT + 1
	pygame.time.set_timer(ALIENLASER,800)

	
	# Build and compile the agent
	actions = env.action_space.n
	model = env.model
	dqn = env.build_agent(model, actions)
	lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
		initial_learning_rate=1e-2, decay_steps=10000, decay_rate=0.9)
	dqn.compile(optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule))
	
	# Training Loop
	total_episodes = 1000
	for episode in range(total_episodes):
		state = env.reset()
		done = False
		total_reward = 0
		
		while not done:
			# Epsilon-greedy action selection
			action = dqn.model.predict(state[None])[0].argmax()
			next_state, reward, done, _ = env.step(action)
			total_reward += reward
			
			# Store the experience in the replay memory
			dqn.remember(state, action, reward, next_state, done)
			state = next_state
			
			# Perform a training update if enough experiences are in the replay memory
			if len(dqn.memory) >= dqn.batch_size:
				dqn.train()

			# Update the target network (if applicable, for DQN)
			dqn.update_target_model()
				
			# Decay epsilon after each episode
			if dqn.policy.__class__.__name__ == 'LinearAnnealedPolicy':
				dqn.policy.eps = max(dqn.policy.eps * dqn.policy.value_decay, dqn.policy.value_min)

			screen.fill((30,30,30))
			env.run()
			#crt.draw()
			
			pygame.display.flip()
			clock.tick(60)
		
		# Print progress after each episode
		print(f"Episode: {episode + 1}/{total_episodes}, Total Reward: {total_reward}")

		
		
	# Save the trained model
	dqn.save_weights('dqn_trained_weights.h5', overwrite=True)