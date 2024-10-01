# Import necessary libraries
import streamlit as st
import pandas as pd
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import dendrogram, linkage

# Function to load and clean data
@st.cache_data
def load_and_clean_data(file_path):
    # Load data
    df_kiva_loans = pd.read_csv(file_path)

    # Clean data
    df_kiva_loans = df_kiva_loans.drop(['use', 'disbursed_time', 'funded_time', 'posted_time', 'tags'], axis=1)
    df_kiva_loans.dropna(subset=['partner_id', 'borrower_genders'], inplace=True)

    # Calculate Z-scores
    z_scores = zscore(df_kiva_loans['funded_amount'])
    df_kiva_loans['outlier_funded_amount'] = (z_scores > 3) | (z_scores < -3)
    df_kiva_loans_cleaned = df_kiva_loans[~df_kiva_loans['outlier_funded_amount']]
    
    return df_kiva_loans_cleaned

# Load the cleaned data
file_path = 'kiva_loans.csv'
df_kiva_loans_cleaned = load_and_clean_data(file_path)

# Streamlit App Title
st.title('BDS24_Weekly_Assignment_Week 2 | Tryfonas Karmiris')

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page:", ["Introduction", "Data Overview", "Top Values by Selected Variable", "Repayment Interval by Selected Variable", "Country Comparison Deepdive", "Sector Comparison Deepdive", "KMeans Clustering & Recommendations","Hierarchical Clustering & Dendrogram"])

# Introduction Page
if page == "Introduction":
    st.subheader("Introduction")
    st.write("""
        This application provides insights into Kiva loans data. 
        You can explore the distribution of funded amounts, 
        analyze top values by selected variables, and visualize 
        relationships between funded amounts and various factors such as Countries and Sectors that the loans were funded.
    """)

# Data Overview Page
elif page == "Data Overview":
    st.subheader("Data Overview")
    st.write("Here is a preview of the cleaned Kiva loans data:")
    
    # Display the cleaned data table
    st.table(df_kiva_loans_cleaned.head())

    # Distribution of Funded Amounts
    st.subheader('Distribution of Funded Amounts')
    chart = alt.Chart(df_kiva_loans_cleaned).mark_bar().encode(
        alt.X('funded_amount', bin=alt.Bin(maxbins=50)),  # Use funded_amount for distribution
        y='count()',
    ).properties(
        title='Distribution of Funded Amounts'
    )
    st.altair_chart(chart, use_container_width=True)
    st.write("This chart shows the distribution of funded amounts for Kiva loans. The x-axis represents the funded amount, while the y-axis shows the count of loans that fall within each bin. As you can see most of the loans are low valued with most of them being in the range of 100 and 500")

# Page 3: Top Values by Selected Variable
elif page == "Top Values by Selected Variable":
    st.subheader('Top Values by Selected Variable')

    # Dropdown for plot type
    plot_type = st.selectbox("Select Variable to Display", ['country', 'repayment_interval', 'sector'])

    # Slider to select the number of top values to display
    num_columns = st.slider(
        "Select Number of Columns to Display on the Chart",
        min_value=5,
        max_value=50,
        value=10,  # default value
        step=1
    )

    # Select the top values based on the selected variable and number of columns
    if plot_type == 'country':
        top_values = df_kiva_loans_cleaned.groupby('country')['funded_amount'].agg(['sum', 'count']).nlargest(num_columns, 'sum').reset_index()
        x_column = 'country'
        count_column = 'count'
        description = f"This chart displays the top {num_columns} countries by total funded amount. The blue bars represent the total funded amount, while the red line indicates the count of loans. In general Phillipines is the country with the most loans followed by Kenya and El Salvador."
    elif plot_type == 'repayment_interval':
        top_values = df_kiva_loans_cleaned.groupby('repayment_interval')['funded_amount'].agg(['sum', 'count']).nlargest(num_columns, 'sum').reset_index()
        x_column = 'repayment_interval'
        count_column = 'count'
        description = f"This chart shows the top {num_columns} repayment intervals by total funded amount. The blue bars represent the total funded amount, while the red line indicates the count of loans. Most of the loans are funded with a monthly repayment interval, where the bullet repayment is an unsusal choice"
    else:  # sector
        top_values = df_kiva_loans_cleaned.groupby('sector')['funded_amount'].agg(['sum', 'count']).nlargest(num_columns, 'sum').reset_index()
        x_column = 'sector'
        count_column = 'count'
        description = f"This chart illustrates the top {num_columns} sectors by total funded amount. The blue bars represent the total funded amount, while the red line indicates the count of loans. Most loans are funded to the Agriculture Sector with Food and Retail completing the first three. Looks like that if the sector of the business is close to Primary production or its Basic Necessities(food) "

    # Display description
    st.write(description)

    # Create a dual-axis bar plot using Matplotlib
    fig, ax1 = plt.subplots(figsize=(12, 9))
    plt.xticks(rotation=90)

    # Bar plot for funded_amount
    color = 'tab:blue'
    ax1.set_xlabel(x_column.replace("_", " ").title())
    ax1.set_ylabel('Funded Amount', color=color)
    ax1.bar(top_values[x_column], top_values['sum'], color=color, alpha=0.6, label='Funded Amount')
    ax1.tick_params(axis='y', labelcolor=color)

    # Create a second y-axis for count
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Count', color=color)
    ax2.plot(top_values[x_column], top_values[count_column], color=color, marker='o', linestyle='-', linewidth=2, label='Count')
    ax2.tick_params(axis='y', labelcolor=color)

    # Add titles and labels
    plt.title(f'Top {num_columns} by {plot_type.replace("_", " ").title()}')
    fig.tight_layout()
    st.pyplot(fig)

    # Boxplot after the dual-axis plot
    st.subheader('Funded Amount vs. Selected Variable')

    # Filter the data based on the selected variable and number of top values
    if plot_type == 'sector':
        top_values_boxplot = df_kiva_loans_cleaned.groupby('sector')['funded_amount'].agg('sum').nlargest(num_columns).index
        filtered_df_boxplot = df_kiva_loans_cleaned[df_kiva_loans_cleaned['sector'].isin(top_values_boxplot)]
    elif plot_type == 'country':
        top_values_boxplot = df_kiva_loans_cleaned.groupby('country')['funded_amount'].agg('sum').nlargest(num_columns).index
        filtered_df_boxplot = df_kiva_loans_cleaned[df_kiva_loans_cleaned['country'].isin(top_values_boxplot)]
    else:  # repayment_interval
        filtered_df_boxplot = df_kiva_loans_cleaned

    # Create a boxplot
    fig, ax = plt.subplots(figsize=(12, 6))
    if plot_type != 'repayment_interval':
        top_values_sorted = df_kiva_loans_cleaned.groupby(plot_type)['funded_amount'].agg('sum').nlargest(num_columns).index
        sns.boxplot(x=plot_type, y='funded_amount', data=filtered_df_boxplot, order=top_values_sorted, ax=ax)
        st.write(f"This boxplot shows the distribution of funded amounts for the top {num_columns} {plot_type.replace('_', ' ')}. It provides insights into the spread and outliers of funded amounts.")
    else:
        sns.boxplot(x=plot_type, y='funded_amount', data=filtered_df_boxplot, ax=ax)
        st.write(f"This boxplot shows the distribution of funded amounts for the top {num_columns} {plot_type.replace('_', ' ')}. It provides insights into the spread and outliers of funded amounts.")

    plt.title('Funded Amount by Selected Variable')
    plt.xlabel(plot_type)
    plt.ylabel('Funded Amount')
    plt.xticks(rotation=90)
    st.pyplot(fig)

# Remaining pages (Repayment Interval by Selected Variable, Country Comparison Deepdive, Sector Comparison Deepdive)
elif page == "Repayment Interval by Selected Variable":
    st.subheader('Repayment Interval by Selected Variable')

    # Dropdown for selecting variable for Seaborn countplot
    plot_var = st.selectbox("Select Variable for Countplot", ['sector', 'country'])

    # Slider to select the number of top values to display for Seaborn countplot
    num_top_values = st.slider(
        "Select Number of Top Values to Display",
        min_value=5,
        max_value=50,
        value=10,  # default value
        step=1
    )

    # Filter the data based on the selected variable and number of top values
    if plot_var == 'sector':
        top_values_plot = df_kiva_loans_cleaned.groupby('sector')['funded_amount'].agg('count').nlargest(num_top_values).index
        filtered_df_plot = df_kiva_loans_cleaned[df_kiva_loans_cleaned['sector'].isin(top_values_plot)]
        description = f"This countplot shows the distribution of repayment intervals for the top {num_top_values} sectors based on the number of loans. In terms of sectors Agriculture got the most monthly repayment loans followed by food. Also a lot of irregulars were in the Food, Retail and Agriculture sectors, which again confirms that loans for first necessities are given more easily. "
    elif plot_var == 'country':
        top_values_plot = df_kiva_loans_cleaned.groupby('country')['funded_amount'].agg('count').nlargest(num_top_values).index
        filtered_df_plot = df_kiva_loans_cleaned[df_kiva_loans_cleaned['country'].isin(top_values_plot)]
        description = f"This countplot illustrates the distribution of repayment intervals for the top {num_top_values} countries based on the number of loans. In terms of countries the Philippines had a great number of Irregular loans."

    # Display description
    st.write(description)

    # Create a count plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Count the occurrences of repayment intervals for the filtered data
    count_data = filtered_df_plot.groupby('repayment_interval')[plot_var].value_counts().unstack(fill_value=0)

    # Calculate total counts for sorting
    total_counts = count_data.sum(axis=1)

    # Sort the repayment intervals based on the total count of loans in descending order
    sorted_index = total_counts.sort_values(ascending=False).index
    count_data = count_data.loc[sorted_index]

    # Create a grouped bar plot
    count_data.plot(kind='bar', ax=ax, position=0, width=0.8)
    plt.title(f'Repayment Interval by {plot_var.replace("_", " ").title()}')
    plt.xlabel('Repayment Interval')
    plt.ylabel('Count of Loans')
    plt.xticks(rotation=45)
    plt.legend(title=plot_var.replace("_", " ").title(), bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(fig)

# Page 5: Country Comparison Deepdive
elif page == "Country Comparison Deepdive":
    st.subheader("Country Comparison Deepdive")

    # Multi-select for countries
    selected_countries = st.multiselect("Select Countries to Compare (Please select one or more)", options=df_kiva_loans_cleaned['country'].unique())

    # Option to choose between count or sum of funded amounts
    aggregation_option = st.radio("Select Aggregation Type:", ("Count of Loans", "Summary of Funded Amount"))

    if selected_countries:
        # Filter the data based on selected countries
        filtered_data = df_kiva_loans_cleaned[df_kiva_loans_cleaned['country'].isin(selected_countries)]

        # Create a combined bar plot for sector summary
        st.subheader("Total Funded Amounts by Sector for Selected Countries")
        if aggregation_option == "Sum":
            sector_summary = filtered_data.groupby(['country', 'sector']).agg(
                total_funded_amount=('funded_amount', 'sum')
            ).reset_index()
            st.write("This graph shows the total funded amount in each Sector for the selected Countries by the user.")
        else:  # Count
            sector_summary = filtered_data.groupby(['country', 'sector']).agg(
                total_funded_amount=('funded_amount', 'count')
            ).reset_index()
            st.write("This graph shows the number of loans in each Sector for the selected Countries by the user.")

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x='sector', y='total_funded_amount', hue='country', data=sector_summary, ax=ax)
        plt.title(f'Total Funded Amount by Sector for Selected Countries ({aggregation_option})')
        plt.xlabel('Sector')
        plt.ylabel('Total Funded Amount' if aggregation_option == "Sum" else 'Count of Loans')
        plt.xticks(rotation=45)
        st.pyplot(fig)

        # Create a combined bar plot for repayment summary
        st.subheader("Total Funded Amounts by Repayment Interval for Selected Countries")
        if aggregation_option == "Summary of Funded Amount":
            repayment_summary = filtered_data.groupby(['country', 'repayment_interval']).agg(
                total_funded_amount=('funded_amount', 'sum')
            ).reset_index()
            st.write("This graph shows the total funded amount in each Repayment interval for the selected Countries by the user.")
        else:  # Count
            repayment_summary = filtered_data.groupby(['country', 'repayment_interval']).agg(
                total_funded_amount=('funded_amount', 'count')
            ).reset_index()
            st.write("This graph shows the number of loans in each Repayment interval for the selected Countries by the user.")

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x='repayment_interval', y='total_funded_amount', hue='country', data=repayment_summary, ax=ax)
        plt.title(f'Total Funded Amount by Repayment Interval for Selected Countries ({aggregation_option})')
        plt.xlabel('Repayment Interval')
        plt.ylabel('Total Funded Amount' if aggregation_option == "Sum" else 'Count of Loans')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.write("Please select one or more countries to compare from the dropdown above.")

# Page 6: Sector Comparison Deepdive
elif page == "Sector Comparison Deepdive":
    st.subheader("Sector Comparison Deepdive")

    # Multi-select for sectors
    selected_sectors = st.multiselect("Select Sectors to Compare (Please select one or more)", options=df_kiva_loans_cleaned['sector'].unique())

    # Option to choose between count or sum of funded amounts
    aggregation_option = st.radio("Select Aggregation Type:", ("Count of Loans", "Summary of Funded Amount"))

    if selected_sectors:
        # Filter the data based on selected sectors
        filtered_data = df_kiva_loans_cleaned[df_kiva_loans_cleaned['sector'].isin(selected_sectors)]

        # Create a combined bar plot for sector summary by country
        st.subheader("Total Funded Amounts by Country for Selected Sectors")
        if aggregation_option == "Summary of Funded Amount":
            country_summary = filtered_data.groupby(['country', 'sector']).agg(
                total_funded_amount=('funded_amount', 'sum')
            ).reset_index()
            st.write("This graph shows the total funded amount in each Country, for the selected Sectors by the user.")
        else:  # Count
            country_summary = filtered_data.groupby(['country', 'sector']).agg(
                total_funded_amount=('funded_amount', 'count')
            ).reset_index()
            st.write("This graph shows the number of loans in each Country, for the selected Sectors by the user.")

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x='country', y='total_funded_amount', hue='sector', data=country_summary, ax=ax)
        plt.title(f'Total Funded Amount by Country for Selected Sectors ({aggregation_option})')
        plt.xlabel('Country')
        plt.ylabel('Total Funded Amount' if aggregation_option == "Sum" else 'Count of Loans')
        plt.legend(title='Sector', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=90)
        st.pyplot(fig)

        # Create a combined bar plot for repayment summary
        st.subheader("Total Funded Amounts by Repayment Interval for Selected Sectors")
        if aggregation_option == "Sum":
            repayment_summary = filtered_data.groupby(['repayment_interval', 'sector']).agg(
                total_funded_amount=('funded_amount', 'sum')
            ).reset_index()
            st.write("This graph shows the funded amount in each Repayment interval for the selected Sectors  by the user.")
        else:  # Count
            repayment_summary = filtered_data.groupby(['repayment_interval', 'sector']).agg(
                total_funded_amount=('funded_amount', 'count')
            ).reset_index()
            st.write("This graph shows the number of loans in each Repayment interval for the selected Sectors by the user.")

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(x='repayment_interval', y='total_funded_amount', hue='sector', data=repayment_summary, ax=ax)
        plt.title(f'Total Funded Amount by Repayment Interval for Selected Sectors ({aggregation_option})')
        plt.xlabel('Repayment Interval')
        plt.ylabel('Total Funded Amount' if aggregation_option == "Sum" else 'Count of Loans')
        plt.legend(title='Sector', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=90)
        st.pyplot(fig)
    else:
        st.write("Please select one or more countries to compare from the dropdown above.")

# Page 7: KMeans Clustering & Recommendations
elif page == "KMeans Clustering & Recommendations":
    st.subheader("KMeans Clustering & Recommendations")

    # User input to choose the number of sample rows
    sample_size = st.slider("Select the number of sample rows for clustering:", min_value=1000, max_value=100000, value=20000, step=1000)

    # Sample the selected number of rows from the DataFrame
    df_sample = df_kiva_loans_cleaned.sample(n=sample_size, random_state=42).copy()

    # Keeping only the relevant columns and storing original indices
    df_original = df_sample[['country','funded_amount', 'sector','repayment_interval']].copy()
    df_original['original_index'] = df_sample.index  # Keep track of original indices

    # Label Encoding for categorical variables and adding encoded columns with "_id" suffix
    label_encoders = {}
    for column in df_original.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df_original[column + '_id'] = le.fit_transform(df_original[column])
        label_encoders[column] = le

    # Standardizing the data using the encoded columns
    encoded_columns = [col + '_id' for col in df_original.select_dtypes(include=['object']).columns]
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_original[encoded_columns + ['funded_amount']])

    # Applying PCA
    pca = PCA(n_components=2)  # Reduce to 2 dimensions for visualization
    df_pca = pca.fit_transform(df_scaled)

    # Elbow Method to find the optimal number of clusters
    inertia = []
    for n in range(1, 11):
        kmeans = KMeans(n_clusters=n, random_state=42)
        kmeans.fit(df_pca)
        inertia.append(kmeans.inertia_)

    # Plotting the Elbow Method
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 11), inertia, marker='o', linestyle='--')
    plt.title('Elbow Method for Optimal Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    st.pyplot(plt.gcf())

    # User input to choose the optimal number of clusters
    optimal_clusters = st.slider("Select the number of optimal clusters:", min_value=1, max_value=10, value=4, step=1)
   
    # Apply KMeans with optimal clusters
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    df_original['cluster'] = kmeans.fit_predict(df_pca)

    # Visualize the clustering results at different iterations
    max_iters = [1, 2, 5, 6, 8, 10]  # Different iterations you want to visualize

    
    plt.figure(figsize=(15, 55))  
    for i, max_iter in enumerate(max_iters, start=1):
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, max_iter=max_iter)
        df_original['cluster'] = kmeans.fit_predict(df_pca)
        
        # Plotting the clusters
        plt.subplot(6, 1, i)  
        sns.scatterplot(x=df_pca[:, 0], y=df_pca[:, 1], hue=df_original['cluster'], palette='viridis', s=100)
        
        # Plotting the centroids
        centroids = kmeans.cluster_centers_
        plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=300, marker='X', label='Centroids')  # Increased centroid size
        
        plt.title(f'K-means Clustering - Iteration {max_iter}', fontsize=16)
        plt.xlabel('Principal Component 1', fontsize=14)
        plt.ylabel('Principal Component 2', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        if i == 1:
            plt.legend()

    plt.tight_layout()
    st.pyplot(plt.gcf())

    # Select a cluster and display top 10 data points
    st.subheader("Explore a Cluster")
    selected_cluster = st.selectbox("Select a Cluster", options=sorted(df_original['cluster'].unique()))

    # Filter data based on selected cluster
    cluster_data = df_original[df_original['cluster'] == selected_cluster]

    st.write(f"Top 10 items in Cluster {selected_cluster}:")
    st.write(cluster_data.head(10))

    # Dynamic input for the new data point
    st.subheader("Input New Data Point for Recommendations")

    # Allow the user to select the country, sector, and repayment interval
    country = st.selectbox("Select Country", options=df_kiva_loans_cleaned['country'].unique())
    sector = st.selectbox("Select Sector", options=df_kiva_loans_cleaned['sector'].unique())
    repayment_interval = st.selectbox("Select Repayment Interval", options=df_kiva_loans_cleaned['repayment_interval'].unique())

    # Allow the user to select the funded amount using a slider
    funded_amount = st.slider("Select Funded Amount", min_value=int(df_kiva_loans_cleaned['funded_amount'].min()), max_value=int(df_kiva_loans_cleaned['funded_amount'].max()), value=1500)

    new_data = {
        'country': country,
        'funded_amount': funded_amount,
        'sector': sector,
        'repayment_interval': repayment_interval
    }

    # Convert new data to DataFrame
    new_data_df = pd.DataFrame([new_data])

    # Encode the new data point and add encoded columns with "_id" suffix
    for column in new_data_df.select_dtypes(include=['object']).columns:
        new_data_df[column + '_id'] = label_encoders[column].transform(new_data_df[column])

    # Standardize the new data using the encoded columns
    new_data_scaled = scaler.transform(new_data_df[[col + '_id' for col in new_data_df.select_dtypes(include=['object']).columns] + ['funded_amount']])

    # Apply PCA to the new data
    new_data_pca = pca.transform(new_data_scaled)

    # Predict the cluster for the new data point
    new_cluster = kmeans.predict(new_data_pca)[0]
    
    st.subheader("Top 5 Similar Items to the Input")
    st.write(f"The new data point belongs to cluster: {new_cluster}")
    
    # Get all data points in the same cluster
    cluster_data = df_original[df_original['cluster'] == new_cluster]

    # Apply the same PCA transformation to the scaled data of the entire cluster
    cluster_data_pca = pca.transform(scaler.transform(cluster_data[encoded_columns + ['funded_amount']]))

    # Calculate the Euclidean distance between the new data point and all points in the same cluster
    distances = cdist(new_data_pca, cluster_data_pca, 'euclidean')[0]

    # Add distances to the cluster data DataFrame
    cluster_data = cluster_data.copy()
    cluster_data['distance'] = distances

    # Sort by distance and select the top 5 closest items
    top_5_recommendations = cluster_data.sort_values('distance').head(5)

    # Retrieve the original rows from the original DataFrame before encoding
    recommended_indices = top_5_recommendations['original_index']
    recommendations = df_kiva_loans_cleaned.loc[recommended_indices]

    # Display the original rows as the top 5 recommendations
    st.write(recommendations)



# Page 8: Hierarchical Clustering & Dendrogram
elif page == "Hierarchical Clustering & Dendrogram":
    st.subheader("Hierarchical Clustering & Dendrogram")

    # User input to choose the number of sample rows
    sample_size = st.slider("Select the number of sample rows for clustering:", min_value=100, max_value=5000, value=150, step=50)

    # User input to choose the number of clusters
    n_clusters = st.slider("Select the number of clusters:", min_value=2, max_value=10, value=4, step=1)

    # Sample the selected number of rows from the DataFrame
    df_sample = df_kiva_loans_cleaned.sample(n=sample_size, random_state=42).copy()

    # Keeping only the relevant columns and storing original indices
    df_original = df_sample[['funded_amount', 'loan_amount']].copy()
    df_original['original_index'] = df_sample.index  # Keep track of original indices

    # Standardizing the data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_original[['funded_amount', 'loan_amount']])

    # Perform Agglomerative Clustering with dynamic n_clusters
    agg_clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    df_original['cluster'] = agg_clustering.fit_predict(df_scaled)

    # Plot the resulting clusters
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=df_original['funded_amount'], y=df_original['loan_amount'], hue=df_original['cluster'], palette='viridis', s=50)
    plt.title(f'Agglomerative Clustering (Hierarchical) Results - {n_clusters} Clusters')
    plt.xlabel('Funded Amount')
    plt.ylabel('Loan Amount')
    st.pyplot(plt.gcf())

    # Dendrogram Visualization
    linked = linkage(df_scaled, method='ward')

    plt.figure(figsize=(10, 7))
    dendrogram(linked,
               orientation='top',
               labels=df_original['original_index'].values,  # Loan IDs as labels
               distance_sort='descending',
               show_leaf_counts=True)
    plt.title('Hierarchical Clustering Dendrogram with Loan IDs')
    plt.xlabel('Loan ID')
    plt.xticks(rotation=90)
    plt.ylabel('Distance')
    st.pyplot(plt.gcf())

