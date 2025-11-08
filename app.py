import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title='One-file Streamlit App', layout='wide')

st.title('One-file Streamlit App — Beginner friendly')
st.write('Upload a CSV or use the sample Iris dataset. Explore data, create simple charts, and train a basic classifier.')

# Sidebar controls
st.sidebar.header('Controls')
use_sample = st.sidebar.checkbox('Use sample Iris dataset', value=True)
uploaded_file = st.sidebar.file_uploader('Upload your CSV file', type=['csv'])

@st.cache_data
def load_sample():
    from sklearn.datasets import load_iris
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    return df

# Load data
if uploaded_file is not None and not use_sample:
    try:
        raw = uploaded_file.read().decode('utf-8')
        df = pd.read_csv(StringIO(raw))
    except Exception as e:
        st.error(f'Error reading CSV: {e}')
        st.stop()
else:
    df = load_sample()

st.subheader('Dataset Preview')
st.dataframe(df.head(50))

# Basic info and summary
with st.expander('Data info and summary'):
    buffer = StringIO()
    df.info(buf=buffer)
    info = buffer.getvalue()
    st.text(info)
    st.write('**Summary statistics:**')
    st.dataframe(df.describe(include='all').T)

# Select numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

st.sidebar.subheader('Visualizations')
plot_type = st.sidebar.selectbox('Plot type', ['Histogram', 'Scatter', 'Correlation heatmap'])

if plot_type == 'Histogram':
    col = st.sidebar.selectbox('Choose numeric column', numeric_cols)
    bins = st.sidebar.slider('Bins', 5, 100, 20)
    fig, ax = plt.subplots()
    ax.hist(df[col].dropna(), bins=bins)
    ax.set_xlabel(col)
    ax.set_ylabel('Count')
    st.pyplot(fig)

elif plot_type == 'Scatter':
    x_col = st.sidebar.selectbox('X axis', numeric_cols, index=0)
    y_col = st.sidebar.selectbox('Y axis', numeric_cols, index=1 if len(numeric_cols)>1 else 0)
    color_by = st.sidebar.selectbox('Color by (optional)', [None] + df.columns.tolist())
    fig, ax = plt.subplots()
    if color_by and color_by in df.columns:
        groups = df.groupby(color_by)
        for name, group in groups:
            ax.scatter(group[x_col], group[y_col], label=str(name), alpha=0.7)
        ax.legend()
    else:
        ax.scatter(df[x_col], df[y_col], alpha=0.7)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    st.pyplot(fig)

else:  # Correlation heatmap
    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, annot=True, fmt='.2f', ax=ax)
    st.pyplot(fig)

# Simple Modeling
st.sidebar.subheader('Simple ML (Classification)')
train_model = st.sidebar.checkbox('Train a simple classifier', value=False)

if train_model:
    st.subheader('Train RandomForestClassifier')
    # Choose target
    target_col = st.selectbox('Select target column (must be categorical / integer labels)', df.columns.tolist(), index=len(df.columns)-1)
    if target_col not in numeric_cols:
        st.warning('Target column is not numeric; attempting to encode automatically if possible.')
    # Feature selection
    feature_cols = st.multiselect('Select feature columns (leave empty to use all numeric)', options=df.columns.tolist(), default=numeric_cols)
    if len(feature_cols) == 0:
        st.error('Please pick at least one feature column.')
    else:
        # Prepare data
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        # Encode non-numeric features
        X = pd.get_dummies(X)
        if y.dtype == 'object':
            y = pd.factorize(y)[0]
        # Drop rows with missing values
        data = pd.concat([X, pd.Series(y, name='target')], axis=1).dropna()
        X = data.drop(columns=['target'])
        y = data['target']

        test_size = st.sidebar.slider('Test set size (%)', 10, 50, 25)
        random_state = st.sidebar.number_input('Random state', value=42)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100.0, random_state=int(random_state))
        n_estimators = st.sidebar.slider('Number of trees', 10, 200, 100)
        clf = RandomForestClassifier(n_estimators=int(n_estimators), random_state=int(random_state))
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        st.write(f'**Accuracy on test set:** {acc:.4f}')
        st.write('**Classification report:**')
        st.text(classification_report(y_test, y_pred))

        st.write('**Confusion matrix:**')
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        st.pyplot(fig)

        # Predict on full dataset and allow download
        if st.button('Generate predictions for full dataset'):
            full_pred = clf.predict(X)
            out = df.copy()
            out['prediction'] = full_pred
            st.dataframe(out.head(50))
            csv = out.to_csv(index=False).encode('utf-8')
            st.download_button('Download predictions as CSV', data=csv, file_name='predictions.csv', mime='text/csv')

st.markdown('---')
st.write('If you want changes (different sample data, regression instead of classification, nicer plots, or a UI tweak), tell me and I will update the app.')
