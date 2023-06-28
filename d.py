import sys

locate_python = sys.exec_prefix

print(locate_python)

color_to_fruit = {
    "red": "Strawberry",
    "yellow": "Banana",
    "green": "Honeydew",
}

z = color_to_fruit.get("greenz", "om")
print(z)