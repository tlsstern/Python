import math
import random

def calculator():
    while True:
        print("\nWhat do you want to do? (input Number)")
        print("1. Addition\t\t\t\t2. Subtraction\t\t3. Division")
        print("4. Multiplication\t\t5. Exponential\t\t6. Root")
        print("7. Factorial\t\t\t8. Geometry\t\t\t9. Chain calculation")
        print("10. Linear Functions\t11. Life Science\t12. Rock Paper Scissors")

        try:
            choice = int(input("Enter your choice: "))
            if choice == 1:
                addition()
            elif choice == 2:
                subtraction()
            elif choice == 3:
                division()
            elif choice == 4:
                multiplication()
            elif choice == 5:
                power()
            elif choice == 6:
                root()
            elif choice == 7:
                factorial()
            elif choice == 8:
                geometry()
            elif choice == 9:
                chain_calculation()
            elif choice == 10:
                linear_functions()
            elif choice == 11:
                life_science()
            elif choice == 12:
                rock_paper_scissors()
            else:
                print("Invalid choice. Please enter a number between 1 and 13.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def addition():
    while True:
        try:
            How_many = int(input("How many numbers do you want to calculate together?\n"))
            break
        except ValueError:
            print("Invalid input. Please enter a number.")
    i = 0
    sum = 0
    while(i < How_many):
        while True:
            try:
                num1 = int(input("Enter a number\n"))
                break
            except ValueError:
                print("Invalid input. Please enter a number.")
        sum += num1
        i += 1
    print("Your sum is", sum)
    play_again()

def subtraction():
    while True:
        try:
            How_many = int(input("How many numbers do you want to calculate together?\n"))
            break
        except ValueError:
            print("Invalid input. Please enter a number.")
    i = 0
    while True:
        try:
            result = int(input("What would you want to be your first number?\n"))
            break
        except ValueError:
            print("Invalid input. Please enter a number.")
    print("Starting with:", result)
    while i < How_many - 1:
        while True:
            try:
                num = int(input("Enter a number to subtract\n"))
                break
            except ValueError:
                print("Invalid input. Please enter a number.")
        result -= num
        i += 1
    print("Your result is", result)
    play_again()

def division():
    while True:
        try:
            How_many = int(input("How many numbers do you want to subtract from the first number?\n"))
            break
        except ValueError:
            print("Invalid input. Please enter a number.")
    i = 0
    while True:
        try:
            result = int(input("What would you want to be your first number?\n"))
            break
        except ValueError:
            print("Invalid input. Please enter a number.")
    print("Starting with:", result)
    while i < How_many:
        while True:
            try:
                num = int(input("Enter a number to divide\n"))
                break
            except ValueError:
                print("Invalid input. Please enter a number.")
        result = result / num
        i += 1
    print("Your result is", result)
    play_again()

def multiplication():
    while True:
        try:
            How_many = int(input("How many numbers do you want to multiply together?\n"))
            break
        except ValueError:
            print("Invalid input. Please enter a number.")
    i = 0
    result = 1
    while(i < How_many):
        while True:
            try:
                num1 = int(input("Enter a number\n"))
                break
            except ValueError:
                print("Invalid input. Please enter a number.")
        result = result * num1
        i += 1
    print("Your result is", result)
    play_again()

def power():
    while True:
        result = 0
        try:
            first_number = int(input("Whats your first number?\n"))
            break
        except ValueError:
            print("Invalid input. Please enter a number.")
    while True:
        try:
            exponent = int(input("Whats your exponent?\n"))
            break
        except ValueError:
            print("Invalid input. Please enter a number.")

    result = first_number ** exponent

    print("Your result is", result)
    play_again()

def root():
    while True:
        result = 0
        try:
            first_number = int(input("Of which number do you want to find out the root?\n"))
            break
        except ValueError:
            print("Invalid input. Please enter a number.")
    result = first_number **(1/2)

    print("Your result is", result)
    play_again()

def factorial():
    while True:
        try:
            num = int(input("Enter a positive integer: "))
            if num < 0:
                raise ValueError  # Raise error for negative input
            break
        except ValueError:
            print("Invalid input. Please enter a non-negative integer.")
    result = math.factorial(num)
    print("The factorial of", num, "is", result)
    play_again()

def geometry():
    while True:
        print("\nGeometry Calculations:")
        print("1. Area")
        print("2. Volume")
        try:
            choice = int(input("Enter your choice 1 or 2: "))
            if choice == 1:
                calculate_area()
            elif choice == 2:
                calculate_volume()
            else:
                print("Invalid choice. Please enter 1 or 2.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def calculate_area():
    while True:
        print("\nWhat Area do you want to calculate?")
        print("1. Circle\t\t2. Square\t\t3. Rectangle")
        print("4. Triangle\t\t5. Trapezoid")
        try:
            choice = int(input("Enter your choice: "))
            if choice == 1:
                area_circle()
            elif choice == 2:
                area_square()
            elif choice == 3:
                area_rectangle()
            elif choice == 4:
                area_triangle()
            elif choice == 5:
                area_trapezoid()
            else:
                print("Invalid choice.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        play_again()

def calculate_volume():
    while True:
        print("\nWhat Volume do you want to calculate?")
        print("1. Ball\t\t2. Cube\t\t3. Cuboid")
        print("4. Cylinder\t\t5. Cone")
        try:
            choice = int(input("Enter your choice: "))
            if choice == 1:
                volume_sphere()
            elif choice == 2:
                volume_cube()
            elif choice == 3:
                volume_cuboid()
            elif choice == 4:
                volume_cylinder()
            elif choice == 5:
                volume_cone()
            else:
                print("Invalid choice.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        play_again()

def area_circle():
    while True:
        try:
            radius = float(input("Enter the radius of the circle: "))
            break
        except ValueError:
            print("Invalid input. Please enter a number.")

    area = math.pi * radius ** 2
    print("The area of the circle is", area)

def area_square():
    while True:
        try:
            side = float(input("Enter the side length of the square: "))
            break
        except ValueError:
            print("Invalid input. Please enter a number.")
    area = side * side
    print("The area of the square is", area)

def area_rectangle():
    while True:
        try:
            length = float(input("Enter the length of the rectangle: "))
            width = float(input("Enter the width of the rectangle: "))
            break
        except ValueError:
            print("Invalid input. Please enter numbers only.")
    area = length * width
    print("The area of the rectangle is", area)

def area_triangle():
    while True:
        try:
            base = float(input("Enter the base of the triangle: "))
            height = float(input("Enter the height of the triangle: "))
            break
        except ValueError:

            print("Invalid input. Please enter numbers only.")
            area = 0.5 * base * height
            print("The area of the triangle is", area)

def area_trapezoid():
    while True:
        try:
            base1 = float(input("Enter the length of the first base: "))
            base2 = float(input("Enter the length of the second base: "))
            height = float(input("Enter the height:"))
            break
        except ValueError:
            print("Invalid input. Please enter numbers only.")
    area = 0.5 * (base1 + base2) * height
    print("The area of the trapezoid is", area)

def volume_sphere():
    while True:
        try:
            radius = float(input("Enter the radius of the sphere: "))
            break
        except ValueError:
            print("Invalid input. Please enter a number.")
    volume = (4/3) * math.pi * radius**3
    print("The volume of the sphere is", volume)

def volume_cube():
    while True:
        try:
            side = float(input("Enter the side length of the cube: "))
            break
        except ValueError:
            print("Invalid input. Please enter a number.")
    volume = side**3
    print("The volume of the cube is", volume)

def volume_cuboid():
    while True:
        try:
            length = float(input("Enter the length of the cuboid: "))
            width = float(input("Enter the width of the cuboid: "))
            height = float(input("Enter the height of the cuboid: "))
            break
        except ValueError:
            print("Invalid input. Please enter numbers only.")
    volume = length * width * height
    print("The volume of the cuboid is", volume)

def volume_cylinder():
    while True:
        try:
            radius = float(input("Enter the radius of the cylinder: "))
            height = float(input("Enter the height of the cylinder: "))
            break
        except ValueError:
            print("Invalid input. Please enter numbers only.")
    volume = math.pi * radius**2 * height
    print("The volume of the cylinder is", volume)

def volume_cone():
    while True:
        try:
            radius = float(input("Enter the radius of the cone: "))
            height = float(input("Enter the height of the cone: "))
            break
        except ValueError:
            print("Invalid input. Please enter numbers only.")
    volume = (1/3) * math.pi * radius**2 * height
    print("The volume of the cone is", volume)

def chain_calculation():
    expression = input("Enter the calculation: ")
    try:
        result = eval(expression)
        print("Result:", result)
    except (SyntaxError, TypeError, ZeroDivisionError):
        print("Invalid expression.")
    play_again()

def linear_functions():
    while True:
        try:
            x = int(input("Enter x:\n"))
            break
        except ValueError:
            print("Please enter a number.\n")

    while True:
        try:
            m = int(input("Enter the slope:\n"))
            break
        except ValueError:
            print("Please enter a number.\n")

    y_values = [0, 1, 2, 3, 4, 5]

    print("| x | y |")
    for y in y_values:
        print(f"| {x + m * y} | {y} |")
    play_again()

def life_science():
    while True:
        print("What do you want to calculate?")
        print("1. BMI\t\t\t\t2. Blood Alcohol")
        print("3. Optimal Bedtime\t4. Calorie Needs")
        try:
            choice = int(input("Enter your choice: "))
            if choice == 1:
                calculate_bmi()
            elif choice == 2:
                calculate_blood_alcohol()
            elif choice == 3:
                calculate_optimal_bedtime()
            elif choice == 4:
                calculate_calorie_needs()
            else:
                print("Invalid choice.")
        except ValueError:
            print("Invalid input. Please enter a number.")
        play_again()

def calculate_bmi():
    while True:
        try:
            weight = float(input("Enter your weight in kilograms: "))
            height = float(input("Enter your height in meters: "))
            break
        except ValueError:
            print("Invalid input. Please enter numbers only.")
    bmi = weight / (height ** 2)
    print("Your BMI is:", bmi)

def calculate_blood_alcohol():
    print("This is a simplified calculation for educational purposes only.")
    while True:
        try:
            num_drinks = int(input("Enter the number of drinks consumed: "))
            weight = float(input("Enter your weight in kilograms: "))
            hours_drinking = float(input("Enter the number of hours spent drinking: "))
            break
        except ValueError:
            print("Invalid input. Please enter numbers only.")

    alcohol_distribution_factor = 0.73
    alcohol_elimination_rate = 0.015

    bac = (num_drinks * 0.6 * 100) / (weight * alcohol_distribution_factor) - (alcohol_elimination_rate * hours_drinking)
    bac = max(0, bac)
    print("Estimated Blood Alcohol Content (BAC):", bac)

def calculate_optimal_bedtime():
    print("This is a general guideline and may not be accurate for everyone.")
    while True:
        try:
            wake_up_time = input("Enter your desired wake-up time (e.g., 7:30 AM): ")
            hours_sleep = int(input("Enter the desired number of sleep cycles (1 cycle is about 90 minutes): "))
            break
        except ValueError:
            print("Invalid input. Please enter a valid time and number.")

    try:
        wake_up_hour = int(wake_up_time.split(':')[0])
        wake_up_minute = int(wake_up_time.split(':')[1].split(' ')[0])
        am_pm = wake_up_time.split(':')[1].split(' ')[1].lower()
        if am_pm == "pm" and wake_up_hour != 12:
            wake_up_hour += 12
        if am_pm == "am" and wake_up_hour == 12:
            wake_up_hour = 0

        total_minutes = (wake_up_hour * 60 + wake_up_minute) - (hours_sleep * 90)
        bedtime_hour = (total_minutes // 60) % 24
        bedtime_minute = total_minutes % 60

        print("Optimal Bedtime:", f"{bedtime_hour:02d}:{bedtime_minute:02d}")
    except (ValueError, IndexError):
        print("Invalid time format.")

def calculate_calorie_needs():
    print("This is a rough estimate and may vary based on individual factors.")
    while True:
        try:
            weight = float(input("Enter your weight in kilograms: "))
            height = float(input("Enter your height in centimeters: "))
            age = int(input("Enter your age in years:"))
            activity_level = int(input("Enter your activity level (1-5, 1 being sedentary, 5 being very active): "))
            break
        except ValueError:
            print("Invalid input. Please enter numbers only.")

    bmr = 10 * weight + 6.25 * height - 5 * age + 5

    activity_factors = [1.2, 1.375, 1.55, 1.725, 1.9]
    activity_factor = activity_factors[activity_level - 1]

    daily_calorie_needs = bmr * activity_factor
    print("Estimated daily calorie needs:", daily_calorie_needs)

def rock_paper_scissors():
    choices = ["rock", "paper", "scissors"]
    user_choice = input("Enter your choice \nrock, paper, or scissors: ").lower()
    computer_choice = random.choice(choices)

    print("You chose:", user_choice)
    print("Computer chose:", computer_choice)

    if user_choice == computer_choice:
        print("It's a tie!")
    elif (user_choice == "rock" and computer_choice == "scissors") or \
            (user_choice == "paper" and computer_choice == "rock") or \
            (user_choice == "scissors" and computer_choice == "paper"):
        print("You win!")
    else:
        print("Computer wins!")
    play_again()

def play_again():
    answer = input("\nDo you want to play again?\nPress 1 if yes\nPress 2 if no\n")
    if answer == "1":
        calculator()
    elif answer == "2":
        exit(1)
    else:
        print("Invalid input. Please enter 1 or 2.")
        play_again()

calculator()