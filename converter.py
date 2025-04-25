import pyfiglet
import math
import cmath
from termcolor import colored

def the_banner():
    banner = pyfiglet.figlet_format("Converter")
    print(colored(banner, 'cyan'))
the_banner()

print("*" * 50)
print("*" * 50)

print("Welcome to the Converter!")
print("*" * 50)
print("Please selct the conversion you want to perform:")
print("*" * 50)
print("1. Meters to Kilometers")
print("2. Kilometers to Meters")
print("3. Grams to Kilograms")
print("4. Degree Celcius to Fahrenheit")
print("5. Fahrenheit to Degree Celcius")

choice = input("Enter your choice (1-5): ")
print("*" * 50)

def meters_to_kilometers():
    meters = int(float(input("Enter the value to convert to kilometers: ")))
    print("Converting meters to kilometers...")
    print(meters/1000)
    


def kilometers_to_meters():
    kilometers = int(float(input("Enter the value to convert to meters: ")))    
    print("Converting kilometers to meters...")
    print(kilometers * 1000)
    

def grams_to_kilograms ():
    grams = int(float(input("Enter the value to convert to kilograms: ")))  
    print("Converting grams to kilograms...")
    print(grams/1000)
    

def kilograms_to_grams ():
    kilograms = int(float(input("Enter the value in to convert to  grams: ")))
    print("Converting kilograms to grams...")
    print(kilograms * 1000)
    

def degree_celcius_to_fahrenheit ():
    celcius = int(float(input("Enter the value to convert to Fahrenheit: ")))
    print("Converting Celcius to Fahrenheit...")
    print((celcius * 9/5) + 32)
    

def fahrenheit_to_degree_celcius ():
    fahrenheit = int(float(input("Enter the value to convert to Celcius: ")))
    celcius = (fahrenheit - 32) * 5/9
    print("Converting Fahrenheit to Celcius...")
    print((fahrenheit - 32) * 5/9)
     
def exit_program ():
    print("Exiting the conversions program...")
    



def main ():
    while True:
        if choice == "1":
            meters_to_kilometers()
        elif choice == "2":
            kilometers_to_meters()
        elif choice == "3":
            grams_to_kilograms()
        elif choice == "4":
            degree_celcius_to_fahrenheit()
        elif choice == "5":
            fahrenheit_to_degree_celcius()
        elif choice =="6":
            exit_program()
            break
        else:
            print("Invalid choice. Please try again.")
            continue

        another_one = input("Do you wanna perform another connversion? Yes or No: ").lower()
        if another_one != "yes":
            print("EXITING THE CONVERTER...")
            break 
        

if __name__ == "__main__":
    main()


        
