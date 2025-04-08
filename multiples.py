import math
import cmath

print("=" *50)
sums = 0

n = int(input("Please type in your number here: "))

print("=" *50)

print("the following is the multiples of 3")
print("=" *50)

for x in range(1,7):
    print(x * 3)
print("=" *50)

print("The following is the multiples of 5")
print("=" *50)
for y in range(1,7):
    print(y * 5)
    
print("=" *50)    
print("beneath is the final calculation for the sum of the multiples of 3 and 5 upto n")   
print("=" *50) 
for i in range(1, n+1):
    if i % 3 == 0 and i % 5 == 0:
        sums += i
print("ThE SuM oF MuLTiPlEs OF 3 and 5\n is: ", sums)

