# Import libraries
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn import tree
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.colors

def monoplot(
        link="https://raw.githubusercontent.com/ianni-phd/Datasets/main/2022-11-12_calciatori_marcatura.csv",
        target = "Classe",
        features = ["Passaggio", "Marcatura", "Tiro", "Tecnica"],
        pair=["Tiro", "Marcatura"]
        # Tiro
):
    df = pd.read_csv(link, sep=';', decimal=',')

    # Make it binary
    dict_2_classes = {"D": "other", "C": "other", "A": "strickers"}
    df[target] = df[target].apply(lambda x: dict_2_classes[x])
    classes_labels = list(pd.factorize(df["Classe"])[1])
    df[target] = pd.factorize(df[target])[0]
    n_classes = len(classes_labels)

    #target = ["Classe"]

    X = df[features].values
    y = df[target].values

    # focus on two features
    X_2_cols = df[pair].values
    print(X_2_cols.shape)
    #X_2_cols

    # MODELING
    clf = DecisionTreeClassifier()                                  # PRIMA PROVA
    # clf = DecisionTreeClassifier(min_samples_leaf=2)              # SECONDA PROVA
    clf = DecisionTreeClassifier(min_samples_leaf=2, max_depth=4)   # TERZA PROVA
    clf.fit(X_2_cols, y)
    y_prev = clf.predict(X_2_cols)
    accuracy = accuracy_score(y_prev, y)
    print("accuracy: ", accuracy)

    # Forecast Rabiot (out of the train!!!)
    print("Rabiot:")
    rabiot_tiro_marcatura = np.array((64, 64)).reshape(1, 2)
    y_prev_rabiot = clf.predict(rabiot_tiro_marcatura)
    print("Features: ", rabiot_tiro_marcatura)
    print("Model prediction: ", y_prev_rabiot)

    cmap_yg_gr_lb = matplotlib.colors.ListedColormap(["yellow", "green", "lightblue"])

    plot_colors = ["ygb" if n_classes == 3 else "yb"][0]
    # plot_step = 0.02

    # plt.figure(figsize=(22,6))
    DecisionBoundaryDisplay.from_estimator(
        clf,
        X_2_cols,
        cmap=cmap_yg_gr_lb,  # plt.cm.YlGn,
        response_method="predict",
        xlabel=pair[0],
        ylabel=pair[1],
    )

    fig = plt.figure(figsize=(25, 20))
    _ = tree.plot_tree(clf, feature_names=["Shot", "Marking"], class_names="Class", filled=True)



def multiplot(
        link = "https://raw.githubusercontent.com/ianni-phd/Datasets/main/2022-11-12_calciatori_marcatura.csv",
        target = "Classe",
        features = ["Passaggio", "Marcatura", "Tiro", "Tecnica"],
        dict_2_classes = {"D": "other", "C": "other", "A": "strickers"}
):

    cmap_yg_gr_lb = matplotlib.colors.ListedColormap(["yellow", "green", "lightblue"])

    # reading the datasdet
    df = pd.read_csv(link, sep=';', decimal=',')

    # defining the classes
    df[target] = df[target].apply(lambda x: dict_2_classes[x])
    classes_labels = list(pd.factorize(df[target])[1])
    df[target] = pd.factorize(df[target])[0]
    n_classes = len(classes_labels)



    X = df[features].values
    y = df[target].values

    # Parameters
    plot_colors = ["ygb" if n_classes == 3 else "yb"][0]

    # Define the size and aspect of the final plot
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    fig.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    # deciding the features to compare
    pairs_to_focus = [["Tiro", "Passaggio"], ["Tiro", "Tecnica"], ["Tiro", "Simpatia"],
                      ["Tiro", "Marcatura"], ["Marcatura", "Passaggio"], ["Tecnica", "Marcatura"]]

    for pairidx, pair in enumerate(pairs_to_focus):
        # We only take the two corresponding features
        X_2_cols = df[pair].values

        # Train and result on train (accuracy)
        clf = DecisionTreeClassifier(max_depth=4).fit(X_2_cols, y)
        y_prev = clf.predict(X_2_cols)
        accuracy = np.round(accuracy_score(y_prev, y), 3)

        # Plot the decision boundary
        ax = plt.subplot(2, 3, pairidx + 1)

        plt.tight_layout(h_pad=1.5, w_pad=1.5, pad=2.5)
        DecisionBoundaryDisplay.from_estimator(
            clf,
            X_2_cols,
            cmap=cmap_yg_gr_lb,
            response_method="predict",
            ax=ax,
            xlabel=pair[0],
            ylabel=pair[1],
        )

        # Plot the training points
        for i, color in zip(range(n_classes), plot_colors):
            idx = np.where(y == i)
            plt.title("Accuracy: " + str(accuracy))
            plt.scatter(
                X_2_cols[idx, 0],
                X_2_cols[idx, 1],
                c=color,
                label=classes_labels[i],
                edgecolor="black",
                s=15,
            )

    plt.suptitle("Decision surface of decision trees trained on pairs of features on the Players dataset")
    plt.legend(loc="lower right", borderpad=0, handletextpad=0)
    _ = plt.axis("tight")


def main():
    monoplot()
    #multiplot()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
    multiplot()

