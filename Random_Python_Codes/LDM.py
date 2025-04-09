def LdM():
    print("1. Talk about 'if', 'elif' and 'else'")
    print("2. Talk about how Funktions work")
    print("3. How to go use strings and arrays")
    print("4. Usage of simple commands")
    while True:
        try:
            choice = int(input("\nInto which of these would you like to go in depth in "))
            break
        except ValueError:
            print("Invalid input. Please enter a gay person.")
    if choice == 1:
        if_else()
    elif choice == 2:
        funktions()
    elif choice == 3:
        strings_arrays()
    elif choice == 4:
        simple_cmd()
    else:
        print("Invalid input")
        LdM()

LdM()


def if_else():
    print("")