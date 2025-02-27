

def box():
    i = 1
    j = 2
    n = 2
    print("Welcome to my Box")
    word = input("Enter a word for the Wordbox?\n")
    print()
    length = len(word)
    reverse_word = "".join(reversed(word))


    while i  < length:
        print(word[i -1], end="")
        i += 1
        if i > length - 1:
            print(word[i-1])
            while j < length:
                print(word[j - 1], end="")
                while n  < length:
                    print(" ", end="")
                    n += 1
                print(reverse_word[j - 1])
                j += 1
                n = 2
                i = 0
            while i < length:
                print(reverse_word[i], end="")
                i += 1

while True:
    box()
    play_again = input("\n\nDo you want to play again? (y/n): ")
    if play_again.lower() != 'y':
        break
