import pygame
import os
import time
import random


#initialize font
pygame.font.init()

width, height = 750, 750
WIN = pygame.display.set_mode((width, height)) #pygame surface
pygame.display.set_caption("Space Shooter Tutorial")

#opponent AI spaceship
RED_SPACESHIP = pygame.image.load(os.path.join("assets", "pixel_ship_red_small.png"))
GREEN_SPACESHIP = pygame.image.load(os.path.join("assets", "pixel_ship_green_small.png"))
BLUE_SPACESHIP = pygame.image.load(os.path.join("assets", "pixel_ship_blue_small.png"))

#AI spaceship
YELLOW_SPACESHIP = pygame.image.load(os.path.join("assets", "pixel_ship_yellow.png"))

#lasers
RED_LASER = pygame.image.load(os.path.join("assets", "pixel_laser_red.png"))
GREEN_LASER = pygame.image.load(os.path.join("assets", "pixel_laser_green.png"))
BLUE_LASER = pygame.image.load(os.path.join("assets", "pixel_laser_blue.png"))
YELLOW_LASER = pygame.image.load(os.path.join("assets", "pixel_laser_yellow.png"))

#background
BG = pygame.transform.scale(pygame.image.load(os.path.join("assets", "background-black.png")), (width, height))

class Laser:
    def __init__(self, x, y, img):
        self.x = x
        self.y = y
        self.img = img
        self.mask = pygame.mask.from_surface(self.img)

    def draw(self, window):
        window.blit(self.img, (self.x, self.y))

    def move(self, vel):
        self.y += vel

    def off_screen(self, height):
        return not(self.y <= height and self.y >= 0)
    
    def collision(self, obj):
        return collide(self, obj)


class Ship:

    COOLDOWN = 30

    def __init__(self, x, y, health=100):
        self.x = x
        self.y = y
        self.health = health
        self.ship_img = None
        self.laser_img = None
        self.lasers = []
        self.cool_down_counter = 0

    def draw(self, window):
        window.blit(self.ship_img, (self.x, self.y))
        for laser in self.lasers:
            laser.draw(window)

    def move_lasers(self, vel, obj):
        self.cooldown()
        for laser in self.lasers:
            laser.move(vel)
            if laser.off_screen (height):
                self.lasers.remove(laser)
            elif laser.collision(obj):
                obj.health -= 10
                self.lasers.remove(laser)

    def cooldown(self):
        if self.cool_down_counter >= self.COOLDOWN:
            self.cool_down_counter = 0
        elif self.cool_down_counter > 0:
            self.cool_down_counter += 1


    def shoot(self):
        if self.cool_down_counter == 0:
            laser = Laser(self.x, self.y, self.laser_img)
            self.lasers.append(laser)
            self.cool_down_counter = 1

    def get_width(self):
        return self.ship_img.get_width()
    
    def get_height(self):
        return self.ship_img.get_height()

class Player(Ship):
    def __init__(self, x, y, health=100):
        super().__init__(x, y, health)
        self.ship_img = YELLOW_SPACESHIP
        self.laser_img = YELLOW_LASER
        self.mask = pygame.mask.from_surface(self.ship_img)
        self.max_health = health

    def move_lasers(self, vel, objs):
        self.cooldown()
        for laser in self.lasers:
            laser.move(vel)
            if laser.off_screen (height):
                self.lasers.remove(laser)
            else:
                for obj in objs:
                    if laser.collision(obj):
                        objs.remove(obj)
                        if laser in self.lasers:
                            self.lasers.remove(laser)

    def draw(self, window):
        super().draw(window)
        self.healthbar(window)

    def healthbar(self, window):
        pygame.draw.rect(window, (255,0,0), (self.x, self.y + self.ship_img.get_height() + 10, self.ship_img.get_width(), 10))
        pygame.draw.rect(window, (0,255,0), (self.x, self.y + self.ship_img.get_height() + 10, self.ship_img.get_width() * (self.health/self.max_health), 10))
        if self.health <= 0:
            self.health = 0


class Enemy(Ship):
    COLOUR_map = {
                "red": (RED_SPACESHIP, RED_LASER),
                "green": (GREEN_SPACESHIP, GREEN_LASER),
                "blue": (BLUE_SPACESHIP, BLUE_LASER)
                }

    def __init__(self, x, y, colour, health=100): 
        super().__init__(x, y, health)
        self.ship_img, self.laser_img = self.COLOUR_map[colour]
        self.mask = pygame.mask.from_surface(self.ship_img)
    
    def shoot(self):
        if self.cool_down_counter == 0:
            laser = Laser(self.x-15, self.y, self.laser_img)
            self.lasers.append(laser)
            self.cool_down_counter = 1

    def move(self, vel):
        self.y += vel

def collide(obj1, obj2):
    offset_x = obj2.x - obj1.x
    offset_y = obj2.y - obj1.y
    return obj1.mask.overlap(obj2.mask, (offset_x, offset_y)) != None

class Game:
    def __init__(self):
        pygame.init()
        self.width, self.height = 750, 750
        self.WIN = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Space Shooter Tutorial")

        # Initialize font
        pygame.font.init()

        # Game variables
        self.run = True
        self.FPS = 60
        self.level = 0
        self.lives = 5
        self.main_font = pygame.font.SysFont("comicsans", 50)
        self.lost_font = pygame.font.SysFont("comicsans", 60)
        self.enemies = []
        self.wave_length = 5
        self.enemy_vel = 1
        self.player_vel = 5
        self.laser_vel = 5
        self.player = Player(300, 630)
        self.clock = pygame.time.Clock()
        self.lost = False
        self.lost_count = 0

    def redraw_window(self):
        self.WIN.blit(BG, (0, 0))

        # Draw text
        lives_label = self.main_font.render(f"Lives: {self.lives}", 1, (255, 255, 255))
        level_label = self.main_font.render(f"Level: {self.level}", 1, (255, 255, 255))

        self.WIN.blit(lives_label, (10, 10))
        self.WIN.blit(level_label, (self.width - level_label.get_width() - 10, 10))

        for enemy in self.enemies:
            enemy.draw(self.WIN)

        self.player.draw(self.WIN)

        if self.lost:
            lost_label = self.lost_font.render("You Lost!!", 1, (255, 255, 255))
            self.WIN.blit(lost_label, (self.width / 2 - lost_label.get_width() / 2, 350))

        pygame.display.update()

    def reset(self):
        self.level = 0
        self.lives = 5
        self.enemies = []
        self.player = Player(300, 630)
        self.lost = False
        self.lost_count = 0

    def main(self):
        while self.run:
            self.clock.tick(self.FPS)
            self.redraw_window()

            if self.player.health <= 0:
                self.lives -= 1
                self.player.health = 100

            if self.lives <= 0:
                self.lost = True
                self.lost_count += 1

            if self.lost:
                if self.lost_count > self.FPS * 3:
                    run = False
                else: continue
        
            if len(self.enemies)==0:
                self.level += 1
                self.wave_length += 5
                for i in range(self.wave_length):
                    enemy = Enemy(random.randrange(50, width-100), random.randrange(-1500*self.level/5, -100), random.choice(["red", "blue", "green"]))
                    self.enemies.append(enemy)

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.run = False

            self.player.healthbar(WIN)
        
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP] and self.player.y - self.player_vel > 0: #UP key gets pressed
                self.player.y -= self.player_vel
            if keys[pygame.K_DOWN] and self.player.y + self.player_vel + self.player.get_height() + 15 < height: #DOWN key gets pressed with wall restrictions
                self.player.y += self.player_vel
            if keys[pygame.K_LEFT] and self.player.x - self.player_vel > 0: #LEFT key gets pressed
                self.player.x -= self.player_vel 
            if keys[pygame.K_RIGHT] and self.player.x + self.player_vel + self.player.get_width() < width: #RIGHT key gets pressed
                self.player.x += self.player_vel
            if keys[pygame.K_SPACE]:
                self.player.shoot()

            for enemy in self.enemies[:]:
                enemy.move(self.enemy_vel)
                enemy.move_lasers(self.laser_vel, self.player)

                if random.randrange(0, 2*30) == 1:
                    enemy.shoot()

                if collide(enemy, self.player):
                    self.player.health -= 10
                    self.enemies.remove(enemy)

                elif enemy.y + enemy.get_height() > height:
                    self.lives -= 1
                    self.enemies.remove(enemy)

            if self.player.health <= 0:
                self.player.health = 100
                self.lives -= 1

            if self.lives <= 0:
                self.lost = True
                self.lost_count += 1
        
            self.player.move_lasers(-self.laser_vel, self.enemies)


            # Call reset function if needed
            if self.lost and self.lost_count > self.FPS * 3:
                self.reset()

        pygame.quit()

    def main_menu(self):
        title_font = pygame.font.SysFont("comicsans", 70)
        run = True
        while run:
            self.WIN.blit(BG, (0, 0))
            title_label = title_font.render("Press the mouse to begin...", 1, (255, 255, 255))
            self.WIN.blit(title_label, (self.width / 2 - title_label.get_width() / 2, 350))
            pygame.display.update()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    self.main()


# Create an instance of the Game class
game = Game()
game.main_menu()
