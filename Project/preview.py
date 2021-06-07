import pandas as pd
import csv

# ub = pd.read_csv("UserBehavior.csv/UserBehavior.csv")
#
# print(ub)
with open("UserBehavior.csv/UserBehavior.csv", 'r') as file:
    reader = csv.reader(file)
    # rows = [row for row in reader]
    # print(rows[0:5])
    for row in reader:
        print(row)