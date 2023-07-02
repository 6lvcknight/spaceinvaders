import pygame
import os
import time
import random

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

def main():
    run = True 
    FPS = 60
    clock = pygame.time.Clock()

    def redraw_window():
        WIN.blit(BG, (0,0))
        pygame.display.update()

    while run:
        clock.tick(FPS)
        redraw_window()

        #if we decide to quit the game
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
main()