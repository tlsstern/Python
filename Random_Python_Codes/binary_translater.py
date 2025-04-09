from unicodedata import decimal

def play_again():
    while True:
        playagain = input("Do you want to play again? [Y/N] ")
        if playagain.upper() == "Y":
            print("\n\n")
            choice()
            break
        elif playagain.upper() == "N":
            exit(1)
        else:
            print("Invalid input. Please enter Y or N.")

def choice():
    print("Do you want to:\n1. Integer to Binary\n2. Binary to Integer\n3. Exit")
    while True:
        try:
            choice = int(input(""))
            if choice == 1:
                decToBin()
                break
            elif choice == 2:
                BinToDec()
                break
            elif choice == 3:
                exit(1)
            else:
                print("Invalid choice")
        except ValueError:
            print("Invalid input. Please enter a number.")

def decToBin():
    i = 0
    binary = []
    try:
        num = int(input("What number do you want to have as Binary? "))
        if num == 0:
            print("The binary of 0 is 0")
            play_again()
        while(num > 0):
            binary.append(num % 2)
            num = num // 2
            i += 1
        binary.reverse()
        print("The binary number is: ", *binary, sep="")
        play_again()
    except ValueError:
        print("Invalid input. Please enter an integer.")
        decToBin()

def BinToDec():
    i = 0
    num = 0
    binary = input("What Binary number would you like as an Integer? ")
    length = len(binary)
    for nums in binary:
        if nums == "0":
            length -= 1
        elif nums == "1":
            num += 2**length
            length -= 1
        else:
            print("This is not a binary number!")
            BinToDec()
            return
    num = num / 2
    print(num)
    play_again()

choice()