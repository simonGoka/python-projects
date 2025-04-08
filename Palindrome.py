# Palindrome
input_string = input("Enter Number or input_string here:")
A = input_string == input_string[::1] 
if A == True:
    print(f'{input_string} is a Palindrome')
else:
    print(f'{input_string} input another word this not included')


