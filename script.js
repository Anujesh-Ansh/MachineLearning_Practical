const subjectData = {
    title: "Sample Subject",
    runningCodes: [
        {
        first: "!pip install numpy pandas scikit-learn matplotlib",
        }
    ],
    questions: [
        {
            question: "1. Apply PCA and SVM on a dataset and create plots based on the analysis.",
            code: `
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

# Apply SVM
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)

# Evaluate the results
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of SVM with PCA:", accuracy)

# Plotting the PCA results
plt.figure(figsize=(8, 6))
for i in np.unique(y):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=data.target_names[i])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Iris Dataset')
plt.legend()
plt.show()

    `,
        },
        {
            question: "2. Apply the K-Nearest Neighbours Classifier on a sample dataset and evaluate results.",
            code: `
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Evaluate the results
print("Classification Report:\n", classification_report(y_test, y_pred))

    `,
        },
        {
            question: "3. Apply Naïve Bayes Classifier on a sample dataset and evaluate results.",
            code: `
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply Naive Bayes Classifier
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)

# Evaluate the results
print("Classification Report:\n", classification_report(y_test, y_pred))

    `,
        },
        {
            question: "4. Implement Simple Linear Regression and Logistic Regression.",
            code: `
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Generate synthetic dataset for linear regression
X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
y = np.array([1.5, 2.5, 3.7, 4.0, 5.5, 6.7])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

# Evaluate the results
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Predictions:", y_pred)

    `,

input: `
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
data = load_iris()
X = data.data[:, :2]  # Using only first two features for simplicity
y = (data.target != 0) * 1  # Convert to binary classification (0 vs 1)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)

# Evaluate the results
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

`,

        },
        {
            question: "5. Write a program to cluster a set of points using K-means.",
            code: `
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic dataset
X = np.array([[1, 2], [1, 4], [1, 0], 
              [4, 2], [4, 4], [4, 0]])

# Apply K-means clustering
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# Plot the results
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s=200, c='red', label='Centroids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-means Clustering')
plt.legend()
plt.show()

    `,

        }
    ]
};




// // Render the running codes and questions with copy buttons
// function renderPageContent() {
//     const runningCodesContainer = document.getElementById("running-codes");
//     const questionsContainer = document.getElementById("questions-container");

//     // Render running codes
//     subjectData.runningCodes.forEach(codeObj => {
//         // Use the specific keys (first, second, third) to create code snippets
//         for (const key in codeObj) {
//             if (codeObj.hasOwnProperty(key)) {
//                 // Create a code snippet with the key as the label
//                 runningCodesContainer.appendChild(createCodeSnippet(codeObj[key], key));
//             }
//         }
//     });

//     // Render questions
//     subjectData.questions.forEach((item, index) => {
//         const questionDiv = document.createElement("div");
//         questionDiv.innerHTML = `<h2>Question ${index + 1}: <span>${item.question}</span></h2>`;
        
//         questionDiv.appendChild(createCodeSnippet(item.code, "lex.l"));
//         questionDiv.appendChild(createCodeSnippet(item.input, "input.txt"));
        
//         questionsContainer.appendChild(questionDiv);
//     });
// }

// // Helper function to create a code snippet element with a copy button
// function createCodeSnippet(content, label) {
//     const container = document.createElement("div");
//     const pre = document.createElement("pre");
//     const codeElem = document.createElement("code");
//     codeElem.textContent = content;

//     const copyBtn = document.createElement("button");
//     copyBtn.textContent = "Copy";
//     copyBtn.className = "copy-btn";
//     copyBtn.onclick = () => handleCopy(content, copyBtn);

//     container.innerHTML = `<strong>${label}:</strong>`;
//     pre.appendChild(codeElem);
//     pre.appendChild(copyBtn);
//     container.appendChild(pre);
    
//     return container;
// }

// // Copy code to clipboard and change button state
// function handleCopy(text, button) {
//     navigator.clipboard.writeText(text).then(() => {
//         button.textContent = "Copied";
//         button.style.backgroundColor = "#3a3d40";
//         setTimeout(() => {
//             button.textContent = "Copy";
//             button.style.backgroundColor = "#238636";
//         }, 2000);
//     }).catch(err => {
//         console.error("Failed to copy text: ", err);
//     });
// }

// // Initialize page content
// document.addEventListener("DOMContentLoaded", renderPageContent);

// Render the running codes and questions with copy buttons
function renderPageContent() {
    const runningCodesContainer = document.getElementById("running-codes");
    const questionsContainer = document.getElementById("questions-container");

    // Render running codes
    subjectData.runningCodes.forEach(codeObj => {
        // Use the specific keys (first, second, third) to create code snippets
        for (const key in codeObj) {
            if (codeObj.hasOwnProperty(key)) {
                // Create a code snippet with the key as the label
                runningCodesContainer.appendChild(createCodeSnippet(codeObj[key], key));
            }
        }
    });

    // Render questions
    subjectData.questions.forEach((item, index) => {
        const questionDiv = document.createElement("div");
        questionDiv.innerHTML = `<h2>Question ${index + 1}: <span>${item.question}</span></h2>`;
        
        questionDiv.appendChild(createCodeSnippet(item.code, "lex.l"));
        questionDiv.appendChild(createCodeSnippet(item.input, "input.txt"));
        
        questionsContainer.appendChild(questionDiv);
    });
}

// Helper function to create a code snippet element with a copy button
function createCodeSnippet(content, label) {
    const container = document.createElement("div");
    const pre = document.createElement("pre");
    const codeElem = document.createElement("code");
    codeElem.textContent = content;

    // Make the whole pre element clickable to copy
    pre.onclick = () => handleCopy(content, null);

    const copyBtn = document.createElement("button");
    copyBtn.textContent = "Copy";
    copyBtn.className = "copy-btn";
    copyBtn.onclick = (event) => {
        event.stopPropagation(); // Prevent the pre click from firing
        handleCopy(content, copyBtn);
    };

    container.innerHTML = `<strong>${label}:</strong>`;
    pre.appendChild(codeElem);
    pre.appendChild(copyBtn);
    container.appendChild(pre);
    
    return container;
}

// Copy code to clipboard and change button state
function handleCopy(text, button) {
    navigator.clipboard.writeText(text).then(() => {
        if (button) {
            button.textContent = "Copied";
            button.style.backgroundColor = "#3a3d40";
            setTimeout(() => {
                button.textContent = "Copy";
                button.style.backgroundColor = "#238636";
            }, 2000);
        }
    }).catch(err => {
        console.error("Failed to copy text: ", err);
    });
}

// Initialize page content
document.addEventListener("DOMContentLoaded", renderPageContent);