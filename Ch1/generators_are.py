def mygen():
    for i in range(3):
        yield f"Hello {i}"
    print("Finished iterating")
    print("Or am i?")
    yield f"I am yielding {__name__}"


gen = mygen()


for i in gen:
    print(i)

for i in range(100):
    # Hit this at i==50 by setting a pdb bp at this line (breakpoint gets number N) 
    # and then using:
    # condition N i==50
    print(f"!!! THIS IS {i=} !!!!!") 

x=15
print("Maybe something goes wrong now")
y = 0/0
print("Can we recover?")
