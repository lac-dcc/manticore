import re

lines = []

with open("logs.txt") as file:
    lines = file.readlines()

for line in lines:
    match = re.search(r".*(\d+).*", line)  # captura o primeiro número da linha
    if match:
        number = int(match.group(1))  # converte para inteiro
        print(f"{number}")



