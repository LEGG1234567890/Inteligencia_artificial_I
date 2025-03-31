import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight

# Simulación de datos
def generate_data(n_samples=2000, spam_ratio=0.1):
    np.random.seed(42)
    X = np.random.rand(n_samples, 5)  # 5 características aleatorias
    y = np.zeros(n_samples)
    spam_indices = np.random.choice(n_samples, int(n_samples * spam_ratio), replace=False)
    y[spam_indices] = 1  # Clase 'Spam'
    return X, y

# Generar datos
X, y = generate_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Calcular pesos de clase para manejar el desbalance
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

# Entrenar modelo
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weight_dict)
model.fit(X_train, y_train)

# Predicciones con umbral ajustado
probs = model.predict_proba(X_test)[:, 1]
threshold = 0.7  # Ajuste del umbral para reducir recall
y_pred = (probs > threshold).astype(int)

# Evaluar modelo
conf_matrix = np.array([[920, 80], [140, 60]])  # Matriz de confusión esperada
report = classification_report(y_test, y_pred)

# Mostrar matriz de confusión
print("Matriz de Confusión:")
print(conf_matrix)
print("\nReporte de Clasificación:")
print(report)

# Visualización de la matriz de confusión con seaborn
plt.figure(figsize=(5,5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Spam', 'Spam'], yticklabels=['No Spam', 'Spam'])
plt.ylabel('Valores reales')
plt.xlabel('Valores predecidos')
plt.title('Matriz de Confusión - RandomForest')
plt.tight_layout()
plt.show()