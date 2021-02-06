A = 1
B = 0
C = 0
for i in range(10):
    A /= 2
    B += A/2
    C += A/2
    print('i ', i, ' A ', A, ' B ', B, ' C ', C, ' sum ', A + B + C)

    B /= 2
    A += B/2
    C += B/2
    print('i ', i, ' A ', A, ' B ', B, ' C ', C, ' sum ', A + B + C)

    C /= 2
    A += C/2
    B += C/2
    print('i ', i, ' A ', A, ' B ', B, ' C ', C, ' sum ', A + B + C)
