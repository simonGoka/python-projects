import pyfiglet
import os
import subprocess
from termcolor import colored

def display_banner():
    
    banner = pyfiglet.figlet_format("Triangle Tree")
    
    print(colored(banner, "cyan"))
    
display_banner()


def triangle_tree():
    for i in range(1, n + 1):
        print("*" * i)

n = int(input("Enter the number of rows: "))
triangle_tree()
