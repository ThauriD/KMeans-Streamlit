#!/usr/bin/env python
# coding: utf-8

# In[3]:
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import umap
import plotly.express as px
import seaborn as sns


@st.cache_data
def load_own(data_default):
              st.write('The data must be a CSV and contain a unique identifier in the first column')
              if uploaded_file is not None and len(uploaded_file.read()) > 1024 * 1024:
                        st.error("Error: File size is too large.")
              elif uploaded_file is None:
                        st.warning("Please upload a file")
              else:
                        try:    
                            df = pd.read_csv(uploaded_file)
                            st.success('File successfully uploaded and processed.')
                        except: 
                            st.error("Error occurred with uploading")
              return df

@st.cache_data
def load_test(data_default):         
              st.write("This Test Data shows the Annual Surface Temperature Change for each country (1961 - 2022)")
              df = pd.read_csv("https://github.com/ThauriD/DemoData/blob/main/Annual_Surface_Temperature_Change.csv?raw=true")
              df.dropna(axis=0,inplace=True)
              df = df[~df.index.duplicated(keep='first')]
              return df

def standardise_data(df):
    scaler = StandardScaler()
    df = df.iloc[:, 0:].values  # Assuming the first column is User_IDs
    df = scaler.fit_transform(df)
    return df


def pca_graph(df):
    pca = PCA()
    pca.fit(df)
    
    #Create Graph
    fig2, ax = plt.subplots()
    explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    ax.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Explained Variance Ratio")
    ax.set_title("PCA - Cumulative Explained Variance")

    return st.pyplot(fig2)

@st.cache_data
def pca_conversion(standardised_df,original_df,num):
        pca = PCA(n_components=num)
        df = pd.DataFrame(data=pca.fit_transform(standardised_df),index=original_df.index)
        return df


@st.cache_resource
def skree_plot(df):
    sse = []
    k_values = range(2, 15)  # Range of cluster numbers to test

    for k in k_values:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(df)
        sse.append(kmeans.inertia_)

    fig3, ax3 = plt.subplots()
    ax3.plot(k_values, sse, marker="o")
    ax3.set_xlabel("Number of Clusters (K)")
    ax3.set_ylabel("Sum of Squared Errors (SSE)")
    ax3.set_title("Elbow Method - Skree Plot")

    return st.pyplot(fig3)

@st.cache_data
def kmeans_clustering(df,clusters):
    kmeans = KMeans(n_clusters=clusters)
    kmeans.fit(df)
    labels = kmeans.labels_
    return labels


@st.cache_resource
def umap_visualisation(neighbours, distance, chart_type, df, clusters):

    if chart_type == '3d':

        umap_class = umap.UMAP(n_neighbors=neighbours,
                               min_dist=distance,
                               spread=1,
                               n_components=3)
        proj = umap_class.fit_transform(df)
        proj = np.column_stack((proj, df.index))

        fig = px.scatter_3d(proj, x=0, y=1, z=2,
                            color=labels.astype(str),
                            labels={'color': 'Cluster', '3': "User_ID"},
                            hover_data=[3])

    if chart_type == '2d':

        umap_class = umap.UMAP(n_neighbors=neighbours,
                               min_dist=distance,
                               spread=1,
                               n_components=2)
        proj = umap_class.fit_transform(df)
        proj = np.column_stack((proj, df.index))

        fig = px.scatter(proj, x=0, y=1,
                         color=labels.astype(str),
                         labels={'color': 'Cluster', '2': "User_ID"},
                         hover_data=[2])
    
    fig.update_traces(marker_size=5)

    return st.plotly_chart(fig)

@st.cache_data
def create_final_df(labels,df):
    df = pd.DataFrame({
                'Clusters': labels
                }, index = df.index).\
                join(df,how="left",sort=False)
            
    return df


#Main Header
st.header('Segmentation Platform')
st.subheader('This program is used to segment data through K-Means Clustering')
st.write("Read this [website](https://towardsdatascience.com/k-means-clustering-and-principal-component-analysis-in-10-minutes-2c5b69c36b6b) to learn about K-Means & PCA")
st.write('Click through the tabs below to progress')


tab1, tab2, tab3, tab4, tab5 = st.tabs(["Upload Data", "EDA", "PCA","K-Means","UMAP Visualisation"])



with tab1:
    st.header("Step 1: Upload File")
    data_default = st.selectbox('Would you like to use test data or upload your own csv?', 
                                ['Test Data','My Data'])
    if data_default == "My Data" :
        uploaded_file = st.file_uploader('Please upload a file to proceed (CSV ONLY)', 
                                         accept_multiple_files=False, 
                                         type=['csv'] )
        df = load_own(data_default)
    elif data_default == "Test Data" :
        df = load_test(data_default)
        
    df = df.set_index(df.columns[0])

    st.write(df)

with tab2:

     # Get the list of column names
    column_names = df.columns    

    col1,col2 = st.columns([1,3])
   
    with col1:
        st.subheader("Histogram")
        # Select the column to view in the bar chart
        column_to_view = st.selectbox('Select a column to view in the bar chart', column_names)
       
    with col2:    
        # Create a bar chart of the selected column
        arr = df[column_to_view]
        fig, ax = plt.subplots()
        ax.margins(x=0.01, y=0.01)
        ax.hist(arr, bins=200)
        ax.set_ylabel('Frequency')
        st.pyplot(fig)


    col1,col2=st.columns([1,3])
    with col1:
        st.subheader("Scatter Plot")
        st.selectbox('Select the two columns to plot in the scatter chart', column_names,key='Col1')
        st.selectbox('Select the two columns to plot in the scatter chart', column_names,key='Col2',label_visibility='hidden')        
    
    with col2:
        fig, ax = plt.subplots()
        ax.margins(x=0.01, y=0.01)
        ax.set_xlabel(st.session_state.Col1)
        ax.set_ylabel(st.session_state.Col2)
        ax.scatter(df[st.session_state.Col1],df[st.session_state.Col2])
        st.pyplot(fig)
        
        
    col1,col2=st.columns([1,3])
    with col1:
        st.subheader("Correlation Heatmap")
        
    with col2:
        corr_df = df.corr()
        fig4, ax4 = plt.subplots()
        sns.heatmap(corr_df,ax=ax4)
        st.pyplot(fig4)
        
        
    col1,col2=st.columns([1,3])
    with col1:
        st.subheader("Correlation Matrix")
        
    with col2:
        st.write(corr_df)   
        




with tab3:
    
    # Standardize the data
    standardised_df = standardise_data(df)

    ## PCA
    st.subheader('Step 2: PCA (Principal Component Analysis). Reduce the dimensionality') 
    st.write("Read this [website](https://towardsdatascience.com/pca-clearly-explained-how-when-why-to-use-it-and-feature-importance-a-guide-in-python-7c274582c37e) to determine when/why to do PCA")
    
    PCA_selected = st.selectbox('Would you like to apply PCA on all columns?', ['Yes','No'])
    
    
    if PCA_selected == 'Yes':
        st.write("Based on the graph below, select the Number of Components that improves explained variance")
        pca_graph(standardised_df)
        
        st.number_input("Input number here",
                        min_value=1,
                        value=1,
                        key = "components")
        st.write("You have chosen ",st.session_state.components,"components in the PCA")
        df1 = pca_conversion(standardised_df, df, st.session_state.components)
        

    else:
        st.write('Skip to Step 3. See Scaled dataframe below')
        df1 = pd.DataFrame(data=standardised_df,index=df.index,columns= df.columns)
    
    st.write(df1)
    



with tab4:
    ## K-Means 
    
    st.subheader('Step 3: K-Means \n Select the number of clusters that maximises the reduction in variance without compromising the efficiency')
    
    # Plotting the skree plot
    skree_plot(df1)
    
    
    st.slider("Input number of clusters here",
              min_value = 1,
              max_value = 15,
              step = 1,
              key = 'clusters')
    st.write("You have chosen ",st.session_state.clusters,"clusters")
    
    # Perform K-means clustering with the selected number of clusters
    labels = kmeans_clustering(df1,st.session_state.clusters)



with tab5:

    #UMAP Visualisation
    st.subheader('Step 4: Visualisation')

    st.selectbox('Would you like to view a 2D or 3D chart?',
                 ['2d', '3d'],
                 key="chart_type")

    st.slider("Input number of neighbours here. Larger values preserve the global view, while smaller values preserve the local views",
              min_value=2,
              max_value=100,
              step=5,
              key='neighbours')

    st.slider("Input distance between points here. Determines how 'clustered' the points are ",
              min_value=0.0,
              max_value=1.0,
              step=0.01,
              key='min_dist')

    umap_visualisation(neighbours=st.session_state.neighbours,
                   distance=st.session_state.min_dist,
                   chart_type=st.session_state.chart_type,
                   df=df1,
                   clusters = st.session_state.clusters)


    #Final Dataframe to download
    final_df = create_final_df(labels, df)
    
    download_button = st.download_button(
        label="Download DataFrame",
        data=final_df.to_csv(index=True),
        file_name="KMEANS-Dataframe.csv",
        mime="text/csv",
        type="primary",
    )
    
    if download_button:
        print("Download button clicked")
        
    st.dataframe(final_df)


#Sidebar
st.sidebar.header("K-Means Clustering App")
st.sidebar.subheader("Step 1: Upload Data // EDA")
st.sidebar.subheader("Step 2: PCA")
st.sidebar.subheader("Step 3: K-Means Clustering")
st.sidebar.subheader("Step 4: UMAP Visualisation",divider='rainbow')

st.sidebar.subheader("Chosen Parameters:")

if PCA_selected == 'Yes':
    st.sidebar.text(f"PCA Components: {st.session_state.components}")
elif PCA_selected == 'No': 
    st.sidebar.text("PCA Components: Not Selected")

st.sidebar.text(f"K-Means Clusters: {st.session_state.clusters}")
