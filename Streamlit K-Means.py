#!/usr/bin/env python
# coding: utf-8

# In[3]:
import streamlit as st
import pandas as pd
import numpy as np
import StandardScaler
import PCA
import KMeans
import matplotlib.pyplot as plt
import umap.umap_ as umap
import plotly.express as px

st.sidebar.header("K-Means Clustering App")
st.sidebar.subheader("Step 1: Input Data")
st.sidebar.subheader("Step 2: PCA")
st.sidebar.subheader("Step 3: K-Means Clustering")
st.sidebar.subheader("Step 4: UMAP Visualisation")

st.header('User Segmentation Platform')
st.write("Read this [website](https://towardsdatascience.com/k-means-clustering-and-principal-component-analysis-in-10-minutes-2c5b69c36b6b) to read about K-Means & PCA")

st.subheader('This program is used to segment users through K-Means Clustering \n Step 1: Upload File. The first column must be called User_IDs and contains unique IDs') 


uploaded_file = st.file_uploader('Upload a file (CSV ONLY)', accept_multiple_files=False, type=['csv'] )

df = pd.read_csv(uploaded_file)
df = df.set_index('User_IDs')

st.write(df)

# Get the list of column names
column_names = df.columns

# Select the column to view in the bar chart
column_to_view = st.selectbox('Select a column to view in the bar chart', column_names)

# Create a bar chart of the selected column
arr = df[column_to_view]
fig, ax = plt.subplots()
ax.margins(x=0.01, y=0.01)
ax.hist(arr, bins=200)
st.pyplot(fig)

# Standardize the data
features = df.iloc[:, 0:].values  # Assuming the first column is User_IDs
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

## PCA

st.subheader('Step 2: PCA (Principal Component Analysis). Reduce the dimensionality') 
st.write("Read this [website](https://towardsdatascience.com/pca-clearly-explained-how-when-why-to-use-it-and-feature-importance-a-guide-in-python-7c274582c37e) to determine when/why to do PCA")

PCA_selected = st.selectbox('Would you like to apply PCA on all columns?', ['Yes','No'])


if PCA_selected == 'Yes':
    st.write("Based on the graph below, select the Number of Components that improves explained variance")
    pca = PCA()
    pca.fit(scaled_features)
    fig2, ax = plt.subplots()
    explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    ax.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Explained Variance Ratio")
    ax.set_title("PCA - Cumulative Explained Variance")

    st.pyplot(fig2)

    
    PCA_num_components = st.number_input("Input number here",min_value=1,value=1)
    st.write("You have chosen ",PCA_num_components,"components in the PCA")
    pca = PCA(n_components=PCA_num_components)
    principal_components = pca.fit_transform(scaled_features)
    df1 = pd.DataFrame(data=principal_components,index=df.index)
    

else:
    st.write('Skip to Step 3. See Scaled dataframe below')
    df1 = pd.DataFrame(data=scaled_features,index=df.index,columns= df.columns)

st.write(df1)

## K-Means 

st.subheader('Step 3: K-Means \n Select the number of clusters that maximises the reduction in variance without compromising the efficiency')


sse = []
k_values = range(2, 15)  # Range of cluster numbers to test

for k in k_values:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(df1)
    sse.append(kmeans.inertia_)


# Plotting the skree plot
fig3, ax3 = plt.subplots()
ax3.plot(k_values, sse, marker="o")
ax3.set_xlabel("Number of Clusters (K)")
ax3.set_ylabel("Sum of Squared Errors (SSE)")
ax3.set_title("Elbow Method - Skree Plot")
st.pyplot(fig3)


KMeans_num_clusters = st.slider("Input number of clusters here",1,10,1)
st.write("You have chosen ",KMeans_num_clusters,"clusters")

# Perform K-means clustering with the selected number of clusters
kmeans = KMeans(n_clusters=KMeans_num_clusters)
kmeans.fit(df1)
labels = kmeans.labels_

#UMAP Visualisation
st.subheader('Step 4: Visualisation')

chart_type = st.selectbox('Would you like to view a 2D or 3D chart?', ['2d', '3d'])

UMAP_n_neighbors = st.slider("Input number of neighbours here. Larger values preserve the global view, while smaller values preserve the local views",2,100,5)
UMAP_min_dist = st.slider("Input distance between points here. Determines how 'clustered' the points are ",0.0,1.0,step=0.01)


if chart_type == '3d':
    
    umap_3d = umap.UMAP(n_neighbors=UMAP_n_neighbors,min_dist=UMAP_min_dist,spread=1,n_components=3)
    proj_3d = umap_3d.fit_transform(df1)
    proj_3d = np.column_stack((proj_3d, df.index))
    
    fig_3d = px.scatter_3d(proj_3d, x=0, y=1, z=2,
        color=labels.astype(str), labels={'color': 'Cluster','3':"User_ID"},hover_data=[3])
    fig_3d.update_traces(marker_size=5)
    st.plotly_chart(fig_3d)

if chart_type == '2d':
    
    umap_2d = umap.UMAP(n_neighbors=UMAP_n_neighbors,min_dist=UMAP_min_dist,spread=1,n_components=2)
    proj_2d = umap_2d.fit_transform(df1)
    proj_2d = np.column_stack((proj_2d, df.index))
    
    fig_2d = px.scatter(proj_2d, x=0, y=1,
       color=labels.astype(str), labels={'color': 'Cluster','2':"User_ID"},hover_data=[2])
    fig_2d.update_traces(marker_size=5)
    st.plotly_chart(fig_2d)



#Final Dataframe to download

labels_df=pd.DataFrame()
labels_df["Clusters"] = pd.DataFrame(labels).set_index(df.index)

combined_df = labels_df.join(df,how="left",sort=False)

download_button = st.download_button(
    label="Download DataFrame",
    data=combined_df.to_csv(index=True),
    file_name="KMEANS-Dataframe.csv",
    mime="text/csv"
)

if download_button:
    print("Download button clicked")


