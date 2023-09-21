#defining the boundaries of the domain
import numpy as np
def decision_subplots(ax, df, classifier, title = "Training Data",steps=0.1):   
    min1, max1 = df.iloc[:,0].min()-1, df.iloc[:,0].max()+1
    min2, max2 = df.iloc[:,1].min()-1, df.iloc[:,1].max()+1
    # print(min1,max1,min2,max2)
    #define all the range
    
    x1grid = np.arange(min1, max1, steps)
    x2grid = np.arange(min2, max2, steps)

    xx, yy = np.meshgrid(x1grid, x2grid)

    r1, r2 = xx.flatten(), yy.flatten()

    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
    grid = np.hstack((r1,r2))
    yhat = classifier.predict(grid)
    zz = yhat.reshape(xx.shape)
    ax.contour(xx, yy, zz, colors='k')
    ax.contourf(xx, yy, zz, cmap='coolwarm', alpha=0.5)
    groups = df.groupby('className')
    for name, group in groups:
        ax.scatter(group.col1, group.col2, marker='o', label=name, alpha=0.4)
    ax.set_title(title)