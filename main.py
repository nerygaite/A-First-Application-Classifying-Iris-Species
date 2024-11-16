from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

iris_dataset = load_iris()

X_train, X_test, y_train, y_test = train_test_split(
    iris_dataset['data'], iris_dataset['target'], random_state=0
)
print("Форма X_train:", X_train.shape)
print("Форма y_train:", y_train.shape)
print("Форма X_test:", X_test.shape)
print("Форма y_test:", y_test.shape)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

print("\nВведіть значення для нової квітки Ірису:")
sepal_length = float(input("Довжина чашолистка (в см): "))
sepal_width = float(input("Ширина чашолистка (в см): "))
petal_length = float(input("Довжина пелюстки (в см): "))
petal_width = float(input("Ширина пелюстки (в см): "))

X_new = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
print("Форма X_new:", X_new.shape)

prediction = knn.predict(X_new)
print("\nПередбачений класс:", prediction)
print("Передбачений вид:", iris_dataset['target_names'][prediction])

y_pred = knn.predict(X_test)
print("Передбачення для тестової вибірки:\n", y_pred)
print("Точність моделі на тестовій вибірці: {:.2f}".format(knn.score(X_test, y_test)))
