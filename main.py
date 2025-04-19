import os

from preprocessing_scripts.preprocessing import conditions, encounters, patients, merge
from synthetic_data_generator.generator import generate
from all_models import train_and_evaluate
from app import interface

os.system("CLS")
choice = 1
while choice != 6:
    print("Welcome to Project for Machine Learning with Big Data".center(80))
    print("".center(80, '='))
    print("You may choose which part of the project you wish to execute here".center(80))
    print("1. Pre-processing")
    print("2. Merging Data")
    print("3. Generate Synthetic Data")
    print("4. Train Models and Output Metrics")
    print("5. View Interface for Predictions")
    print("6. Exit")
    choice = int(input("Enter your choice here: "))
    print("".center(80, '='))

    if choice == '1':
        print("Running Pre-processing...")
        print("Preprocessing conditions.csv")
        conditions()
        print("Preprocessing encounters.csv")
        encounters()
        print("Preprocessing patients.csv")
        patients()
    elif choice == '2':
        print("Running Data Merging...")
        merge()
    elif choice == '3':
        print("Generating Synthetic Data...")
        generate()
    elif choice == '4':
        print("Training Models and Outputting Metrics...")
        train_and_evaluate()
    elif choice == '5':
        interface()
    elif choice == '6':
        print("Exiting Project...")
        break
    else:
        print("Invalid choice. Try again.")