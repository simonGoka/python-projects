#Assignment1

# get user input
start = int(input("enter number:"))
end = int(input("enter number"))

# start number should be less or equal to end
if start > end:
    start, end = end, start
    
# calculation
even_sum = 0

#navigation
for num in range(start, end + 1):
    if num % 2 == 0:
        even_sum += num
        
        
# display the result
print(f"The sum {start} and {end} is: {even_sum}")