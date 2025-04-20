import os

from preprocessing_scripts.preprocessing import conditions, encounters, patients, merge, label_dropoffs_by_kmeans
from synthetic_data_generator.generator import generate
from all_models import train_spark
from model_metrics import evaluate_model
from app import interface

def clrscr():
    try:
        os.system('CLS')
    except:
        os.system('clear')

app_called = False

clrscr()
choice = 1
while choice != 8:
    print("Welcome to Project for Machine Learning with Big Data".center(80))
    print("".center(80, '='))
    print("You may choose which part of the project you wish to execute here".center(80))
    print("1. Pre-processing")
    print("2. Merging Data")
    print("3. Feature Engineering the DROP_OFF")
    print("4. Generate Synthetic Data")
    print("5. Train Models")
    print("6. Output Metrics")
    print("7. View Interface for Predictions")
    print("8. Exit")
    choice = int(input("Enter your choice here: "))
    print("".center(80, '='))

    if choice == 1:
        print("Running Pre-processing...")
        print("Preprocessing conditions.csv")
        conditions()
        print("Preprocessing encounters.csv")
        encounters()
        print("Preprocessing patients.csv")
        patients()
        input("Press enter to continue...")
        clrscr()
    elif choice == 2:
        print("Running Data Merging...")
        merge()
        input("Press enter to continue...")
        clrscr()
    elif choice == 3:
        print("Labelling Data Using K-Means Clustering...")
        label_dropoffs_by_kmeans()
        input("Press enter to continue...")
        clrscr()
    elif choice == 4:
        print("Generating Synthetic Data...")
        generate()
        input("Press enter to continue...")
        clrscr()
    elif choice == 5:
        print("Training Models...")
        train_spark()
        input("Press enter to continue...")
    elif choice == 6:
        print("Outputting Metrics...")
        evaluate_model()
        input("Press enter to continue...")
        clrscr()
    elif choice == 7:
        app_called = True
        break
    elif choice == 8:
        print("Exiting Project...")
        break
    else:
        print("Invalid choice. Try again.")

if app_called:
    interface()