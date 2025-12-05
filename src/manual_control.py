import pygame
import numpy as np

from WorldModel import describe_action_space

def register_input(envs):
    restart, quit = False, False
    a = describe_action_space(envs.single_action_space)["low"]  # Valid default value

    match envs.spec.id:
        case "CarRacing-v3":
    
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        a[0] = -1.0
                    if event.key == pygame.K_RIGHT:
                        a[0] = +1.0
                    if event.key == pygame.K_UP:
                        a[1] = +1.0
                    if event.key == pygame.K_DOWN:
                        a[2] = +0.8  # set 1.0 for wheels to block to zero rotation
                    if event.key == pygame.K_RETURN:
                        restart = True
                    if event.key == pygame.K_ESCAPE:
                        quit = True

                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_LEFT:
                        a[0] = 0
                    if event.key == pygame.K_RIGHT:
                        a[0] = 0
                    if event.key == pygame.K_UP:
                        a[1] = 0
                    if event.key == pygame.K_DOWN:
                        a[2] = 0
            return a, restart, quit

        case "CartPole-v1":
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        a[0] = 0
                    if event.key == pygame.K_RIGHT:
                        a[0] = 1
                    if event.key == pygame.K_RETURN:
                        restart = True
                    if event.key == pygame.K_ESCAPE:
                        quit = True

                if event.type == pygame.KEYUP:
                    if event.key == pygame.K_LEFT:
                        a[0] = 0.5
                    if event.key == pygame.K_RIGHT:
                        a[0] = 0.5
            return a, restart, quit
            
        case _:
            raise Exception("Unknown environment.")
