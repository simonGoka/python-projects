import math
import cmath
import pyfiglet
import os
import subprocess
from termcolor import colored

def display_banner():
    
    banner = pyfiglet.figlet_format("PaSsWoRd ChEcKeR")
    
    print(colored(banner, "cyan"))
    
display_banner()


pass_text = "Python123"

new_entry_password = int(input("Please Enter your Password here: "))

if new_entry_password == int(pass_text):
    print(colored("That is your Correct Password! it is Validated!", "green")) 
else:
    print(colored("That is Incorrect! Not Valid!", "red"))

