# ==============================
# Basic Statistics
# ==============================

def mean(data):
    return sum(data) / len(data)

def variance(data):
    m = mean(data)
    return sum((x - m) ** 2 for x in data) / len(data)

def std_dev(data):
    return variance(data) ** 0.5

def correlation(x, y):
    mx = mean(x)
    my = mean(y)
    numerator = sum((x[i] - mx) * (y[i] - my) for i in range(len(x)))
    denominator = (sum((x[i] - mx) ** 2 for i in range(len(x))) *
                   sum((y[i] - my) ** 2 for i in range(len(y)))) ** 0.5
    return numerator / denominator


# ==============================
# Data Scaling
# ==============================

def z_score(data):
    m = mean(data)
    s = std_dev(data)
    return [(x - m) / s for x in data]

def min_max_scale(data):
    min_val = min(data)
    max_val = max(data)
    return [(x - min_val) / (max_val - min_val) for x in data]


# ==============================
# Probability & Bayes
# ==============================

def probability(event_count, total_count):
    return event_count / total_count

def bayes(prior, likelihood, evidence):
    # P(A|B) = (P(B|A) * P(A)) / P(B)
    return (likelihood * prior) / evidence


# ==============================
# Matrix Operations
# ==============================

def transpose(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    return [[matrix[i][j] for i in range(rows)] for j in range(cols)]

def matrix_multiply(A, B):
    result = []
    for i in range(len(A)):
        row = []
        for j in range(len(B[0])):
            sum_val = 0
            for k in range(len(B)):
                sum_val += A[i][k] * B[k][j]
            row.append(sum_val)
        result.append(row)
    return result


# ==============================
# ML Metrics
# ==============================

def mse(y_true, y_pred):
    return sum((y_true[i] - y_pred[i]) ** 2 for i in range(len(y_true))) / len(y_true)

def sigmoid(x):
    # Using e ≈ 2.71828 (no math library)
    e = 2.718281828
    return 1 / (1 + e ** (-x))

def accuracy(y_true, y_pred):
    correct = sum(1 for i in range(len(y_true)) if y_true[i] == y_pred[i])
    return correct / len(y_true)

def precision(y_true, y_pred):
    tp = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 1)
    fp = sum(1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 1)
    return tp / (tp + fp) if (tp + fp) != 0 else 0

def recall(y_true, y_pred):
    tp = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 1)
    fn = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 0)
    return tp / (tp + fn) if (tp + fn) != 0 else 0

def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) != 0 else 0


# ==============================
# Entropy
# ==============================

def entropy(probabilities):
    e = 2.718281828
    total = 0
    for p in probabilities:
        if p > 0:
            # log base e approximation using change of base:
            # ln(p) ≈ (p - 1) - (p - 1)^2/2  (simple approximation)
            ln_p = (p - 1) - ((p - 1) ** 2) / 2
            total += p * ln_p
    return -total


# ==============================
# Example Usage
# ==============================

data1 = [1, 2, 3, 4, 5]
data2 = [2, 4, 6, 8, 10]

print("Mean:", mean(data1))
print("Variance:", variance(data1))
print("Std Dev:", std_dev(data1))
print("Correlation:", correlation(data1, data2))
print("Z Score:", z_score(data1))
print("Min-Max Scale:", min_max_scale(data1))
print("Probability:", probability(3, 10))
print("Bayes:", bayes(0.5, 0.8, 0.6))

A = [[1, 2], [3, 4]]
B = [[5, 6], [7, 8]]

print("Transpose:", transpose(A))
print("Matrix Multiply:", matrix_multiply(A, B))

y_true = [1, 0, 1, 1, 0]
y_pred = [1, 0, 0, 1, 1]

print("MSE:", mse(y_true, y_pred))
print("Sigmoid(2):", sigmoid(2))
print("Accuracy:", accuracy(y_true, y_pred))
print("Precision:", precision(y_true, y_pred))
print("Recall:", recall(y_true, y_pred))
print("F1 Score:", f1_score(y_true, y_pred))
print("Entropy:", entropy([0.5, 0.5]))