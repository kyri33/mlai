result = [0,0,0,0]
train = 4

def hidden_layer_y(epoch, x1, x2, w1, w2, w3, w4, b1, b2, pred, result):
    h1 = (x1 * w1) + (x2 * w4)
    h2 = (x1 * w2) + (x2 * w3)

    if (h1 >= 1): h1 = 1
    if (h1 < 1): h1 = 0
    if (h2 >=1): h2 = 1
    if (h2 < 1): h2 = 0

    h1 = h1 * -b1
    h2 = h2 * b2

    y = h1 + h2
    if (y < 1 and pred >= 0 and pred < 2):
        result[pred] = 1

    if (y >= 1 and pred >= 2 and pred < 4):
        result[pred] = 1

w1 = 0.5; w2 = 0.5; b1 = 0.5
w3 = w2; w4 = w1; b2 = b1

for epoch in range(50):
    if (epoch < 1):
        w1 = 0.5; w2 = 0.5; b1 = 0.5
    w3 = w2; w4 = w1; b2 = b1
    
    for t in range(4):
        if (t == 0): x1 = 1; x2 = 1
        if (t == 1): x1 = 0; x2 = 0
        if (t == 2): x1 = 1; x2 = 0
        if (t == 3): x1 = 0; x2 = 1
        pred = t
        hidden_layer_y(epoch, x1, x2, w1, w2, w3, w4, b1, b2, pred, result)
    
    print("epoch:",epoch,"optimization",round(train-sum(result)),"w1:",round(w1,4),"w2:",round(w2,4),"w3:",round(w3,4),"w4:",round(w4,4),"b1:",round(-b1,4),"b2:",round(b2,4))
    convergence = sum(result) - train
    if (convergence >= -0.0000000001): break
    else: w2 += 0.05; b1 = w2
    result[0] = 0; result[1] = 0; result[2] = 0; result[3] = 0
    
    #print(convergence)

print("epoch: ",epoch, "w1: " , w1, "w2: ", w2, "b1: ", b1, "b2: ", b2)