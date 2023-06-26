potential = []
for i in range(16):
    for j in range(16):
        potential.append(f"{0.5*(i-8)**2 + 0.5*((j)-8)**2}")

with open("example_pot.in", "w") as file:
    file.writelines(line + '\n' for line in potential)


#f = open("example_pot.in", "w")
#f.writelines(["See you soon!", "asdsadad and out."])
#f.close()
