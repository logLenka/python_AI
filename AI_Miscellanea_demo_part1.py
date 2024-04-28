
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def assignment(df, C, cols):
    d0 = np.sqrt((np.array(df['x']) - C[0, 0]) ** 2 + (np.array(df['y']) - C[0, 1]) ** 2)
    d1 = np.sqrt((np.array(df['x']) - C[1, 0]) ** 2 + (np.array(df['y']) - C[1, 1]) ** 2)
    d2 = np.sqrt((np.array(df['x']) - C[2, 0]) ** 2 + (np.array(df['y']) - C[2, 1]) ** 2)

    d = np.vstack((d0, d1, d2))
    cl = d.argmin(axis=0)
    df['closest'] = cl

    df['color'] = df['closest'].map(lambda x: cols[x])
    return df

def update(C):
    k = np.shape(C)[0]
    for i in range(k):
        C[i,0] = np.mean(df[df['closest'] == i]['x'])
        C[i,1] = np.mean(df[df['closest'] == i]['y'])
    return C

df = pd.DataFrame({
'x': [12, 20, 28, 18, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72],
'y': [39, 36, 30, 52, 54, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24]
})

k = 3
cols = {0: 'r', 1: 'g', 2: 'b'}

np.random.seed(200)

C = np.zeros((k,2),dtype="int64")
for i in range(k):
    C[i,0] = np.random.randint(0, 80)
    C[i,1] = np.random.randint(0, 80)
print(C)

plt.scatter(df['x'], df['y'], color='k')
for i in range(k):
    plt.scatter(C[i,0],C[i,1],color=cols[i],marker="+")
plt.xlim(0, 80)
plt.ylim(0, 80)
plt.title("Initial state")
plt.show()

df = assignment(df,C,cols)
print(df)

plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
for i in range(k):
    plt.scatter(C[i,0],C[i,1], color=cols[i],marker="+")
plt.xlim(0, 80)
plt.ylim(0, 80)
plt.title("colors assigned")
plt.show()


C = update(C)

plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
for i in range(k):
    plt.scatter(C[i,0],C[i,1], color=cols[i],marker="+")
plt.xlim(0, 80)
plt.ylim(0, 80)
plt.title("centroids updated")
plt.show()

for ii in [0,1]:
    df = assignment(df, C,cols)
    C = update(C)

    plt.scatter(df['x'], df['y'], color=df['color'], alpha=0.5, edgecolor='k')
    for i in range(k):
        plt.scatter(C[i,0],C[i,1], color=cols[i],marker="+")
    plt.xlim(0, 80)
    plt.ylim(0, 80)
    plt.title("colors assigned")
    plt.show()
