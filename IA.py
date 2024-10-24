import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Carrega o dataset CIFAR-10, sem usar pandas
cifar10 = fetch_openml('cifar_10', version=1, cache=True, as_frame=False, parser='liac-arff')

# Separa os dados e os rótulos
X = cifar10['data']
y = cifar10['target']

# Converte os rótulos para inteiros
y = y.astype(np.uint8)

# Reduz o dataset para 2.000 exemplos para evitar problemas de lentidão
X = X[:2000]
y = y[:2000]

# Divide os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normaliza os dados
X_train = X_train / 255.0
X_test = X_test / 255.0

# Função para treinar e avaliar um classificador
def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calcula as métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    
    return accuracy, precision, recall, f1

# Modelos com otimizações para reduzir o tempo de execução
models = {
    "SVM (Linear Kernel)": SVC(kernel='linear'),
    "KNN": KNeighborsClassifier(),
    "Random Forest (10 trees)": RandomForestClassifier(n_estimators=10)
}

# Dicionário para armazenar os resultados
results = {
    "Model": [],
    "Accuracy": [],
    "Precision": [],
    "Recall": [],
    "F1 Score": []
}

# Treinamento e avaliação
for name, model in models.items():
    accuracy, precision, recall, f1 = train_and_evaluate_model(model, X_train, X_test, y_train, y_test)
    
    # Armazena os resultados no dicionário
    results["Model"].append(name)
    results["Accuracy"].append(accuracy)
    results["Precision"].append(precision)
    results["Recall"].append(recall)
    results["F1 Score"].append(f1)

# Converte os resultados em um DataFrame pandas
df_results = pd.DataFrame(results)

# Exibe a tabela
print(df_results)
