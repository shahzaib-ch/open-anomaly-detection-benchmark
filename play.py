from sklearn.metrics import f1_score

l = [0, 1, 0]
l2 = [0.0000, 1.000, 1.000]

m = f1_score(l, l2)
print(m)
