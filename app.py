import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go


df_vehicles = pd.read_csv("Datasets/vehicles_us.csv", sep=',', header='infer')

st.header('Web Dashboard with Streamlit and Python for Vehicle sales analysis descriptive statistics')

st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
    .stTabs {
      overflow-x: auto;
      white-space: nowrap;
    }
    .stTabs button {
      display: inline-block;
      white-space: normal;
    }
    </style>
    """,
    unsafe_allow_html=True
)

tab_titles = [
    "1. Distribution of Prices and Mileage",
    "2. Comparison by vehicle type or condition",
    "3. Scatter plots for price vs. model year or mileage",
    "4. Price Trends Over Time",
    "5. Condition Impact",
    "6. Brand vs. Price",
    "7. Mileage vs. Price (Depreciation)",
    "8. Top Models Sales",
    "9. Transmission Type Analysis",
    "10. Fuel Type Preference",
    "11. Correlation Analysis of Vehicle Features"
]

tabs = st.tabs(tab_titles)


for i, tab in enumerate(tabs):
    with tab:
        if i == 0:
            # 1. Descriptive statistics for Distribution of prices and mileage

            # Use session_state to maintain the button state
            if "show_hist" not in st.session_state:
                st.session_state.show_hist = False

            if st.button("Show Distribution of Prices and Mileage"):
                st.session_state.show_hist = True

            if st.session_state.show_hist:
                st.write('Overview of Price and Mileage Distribution')

                # Selector Default: index=2 'Both'
                view_mode = st.radio(
                    "View:",
                    options=['Price only', 'Odometer only', 'Both'],
                    index=2,
                    key="agg_price_milage_dist"
                )

                # Graph according selector
                if view_mode == 'Price only':
                    fig = px.histogram(df_vehicles, x='price', nbins=50, title='Price Distribution',
                                       color_discrete_sequence=['grey'])
                    st.plotly_chart(fig)

                    price_data = df_vehicles['price'].dropna()
                    st.markdown(f"""
                        **Price Insights:**
                        - **Mean:** ${price_data.mean():,.0f}
                        - **Median:** ${price_data.median():,.0f}
                        - **Min:** ${price_data.min():,.0f}
                        - **Max:** ${price_data.max():,.0f}
                    """)

                elif view_mode == 'Odometer only':
                    fig = px.histogram(df_vehicles, x='odometer', nbins=50, title='Odometer Distribution',
                                       color_discrete_sequence=['grey'])
                    st.plotly_chart(fig)

                    odo_data = df_vehicles['odometer'].dropna()
                    st.markdown(f"""
                        **Mileage Insights:**
                        - **Mean:** {odo_data.mean():,.0f} miles
                        - **Median:** {odo_data.median():,.0f} miles
                        - **Min:** {odo_data.min():,.0f} miles
                        - **Max:** {odo_data.max():,.0f} miles
                    """)

                else:
                    fig_price = px.histogram(df_vehicles, x='price', nbins=50,
                                             color_discrete_sequence=['grey'])
                    fig_odometer = px.histogram(df_vehicles, x='odometer', nbins=50,
                                                color_discrete_sequence=['grey'])

                    fig = make_subplots(
                        rows=1, cols=2, subplot_titles=("Price", "Odometer"))
                    fig.add_trace(fig_price.data[0], row=1, col=1)
                    fig.add_trace(fig_odometer.data[0], row=1, col=2)
                    fig.update_layout(
                        title_text="Distribution of Price and Odometer", showlegend=False)

                    st.plotly_chart(fig)

                    # Both insights
                    price_data = df_vehicles['price'].dropna()
                    odo_data = df_vehicles['odometer'].dropna()

                    st.subheader("Price Insights")
                    st.markdown(f"""
                        - **Mean:** ${price_data.mean():,.0f}
                        - **Median:** ${price_data.median():,.0f}
                        - **Min:** ${price_data.min():,.0f}
                        - **Max:** ${price_data.max():,.0f}
                    """)

                    st.subheader("Mileage Insights")
                    st.markdown(f"""
                        - **Mean:** {odo_data.mean():,.0f} miles
                        - **Median:** {odo_data.median():,.0f} miles
                        - **Min:** {odo_data.min():,.0f} miles
                        - **Max:** {odo_data.max():,.0f} miles
                    """)

        if i == 1:
            # 2. Descriptive statistics for Comparison by vehicle type or condition

            # Use session_state to maintain the button state
            if "show_bar" not in st.session_state:
                st.session_state.show_bar = False

            # Button for Bar Plot
            if st.button("Show Bar Plot for Comparison by Vehicle Type or Condition"):
                st.session_state.show_bar = True

            if st.session_state.show_bar:
                st.write('Vehicle type or condition comparison')

                # Interactive dropdown
                compare_option = st.selectbox(
                    "Compare by:",
                    options=[('Vehicle Type', 'type'),
                             ('Condition', 'condition')],
                    format_func=lambda x: x[0],
                    key="agg_vehicle_condition"
                )

                # Get column name
                group_by_col = compare_option[1]

                # Group and count
                grouped = df_vehicles[group_by_col].value_counts(
                ).reset_index()
                grouped.columns = [group_by_col, 'count']

                # Bar plot
                fig = px.bar(grouped, x=group_by_col, y='count',
                             title=f'Number of Listings by {group_by_col.capitalize()}',
                             labels={group_by_col: group_by_col.capitalize(),
                                     'count': 'Listings'},
                             color_discrete_sequence=['grey'])

                fig.update_layout(xaxis={'categoryorder': 'total descending'})

                st.plotly_chart(fig)

                # Insights
                most_common = grouped.iloc[0]
                least_common = grouped.iloc[-1]
                total = grouped['count'].sum()

                st.markdown(f"""
                **Insights for {group_by_col.capitalize()}:**
                - **Most frequent:** {most_common[group_by_col]} ({most_common['count']} listings)
                - **Least frequent:** {least_common[group_by_col]} ({least_common['count']} listings)
                - **Total listings shown:** {total}
                """)

        if i == 2:
            # 3. Descriptive statistics for Scatter plots for price vs. model year or mileage

            # Use session_state to maintain the button state
            if "show_scatter" not in st.session_state:
                st.session_state.show_scatter = False

            # Button for Scatter Plot
            if st.button("Show Scatter Plot for Price vs. Model, Year or Mileage"):
                st.session_state.show_scatter = True

            if st.session_state.show_scatter:
                st.write('Relationship between price vs. Model, Year or Mileage')

                # Interactive Dropdown
                xaxis_option = st.selectbox(
                    "X-axis:",
                    options=[('Model Year', 'model_year'),
                             ('Mileage (Odometer)', 'odometer')],
                    format_func=lambda x: x[0],
                    key="agg_price_model_milage"
                )

                x_axis = xaxis_option[1]
                df_plot = df_vehicles[[x_axis, 'price',
                                       'model', 'condition', 'fuel']].dropna()

                # Scatter Plot
                fig = px.scatter(df_vehicles, x=x_axis, y='price',
                                 title=f'Price vs. {x_axis.replace("_", " ").title()}',
                                 labels={x_axis: x_axis.replace(
                                     "_", " ").title(), 'price': 'Price'},
                                 opacity=0.6,
                                 hover_data=['model', 'condition', 'fuel'],
                                 color_discrete_sequence=['grey'])

                st.plotly_chart(fig)

                # Insights
                st.markdown(f"""
                **Insights for {x_axis.replace('_', ' ').title()}:**
                - **Min {x_axis}:** {df_plot[x_axis].min():,.0f}
                - **Max {x_axis}:** {df_plot[x_axis].max():,.0f}
                - **Mean {x_axis}:** {df_plot[x_axis].mean():,.0f}
                - **Median {x_axis}:** {df_plot[x_axis].median():,.0f}
                - **Average price (filtered):** ${df_plot['price'].mean():,.0f}
                """)

        if i == 3:
            # 4. Descriptive statistics for Price Trends Over Time

            # Use session_state to maintain the button state
            if "show_line" not in st.session_state:
                st.session_state.show_line = False

            # Button for Line chart
            if st.button("Show Line Chart for Price Trends Over Time"):
                st.session_state.show_line = True

            if st.session_state.show_line:
                st.write('Average or median price trend by model year')

                # Interactive Dropdown
                agg_option = st.selectbox(
                    "Aggregate:",
                    options=[('Mean', 'mean'), ('Median', 'median')],
                    format_func=lambda x: x[0],
                    key="agg_price_trends",
                )

                agg_func = agg_option[1]

                # Filter null values
                df_filtered = df_vehicles.dropna(
                    subset=['model_year', 'price'])

                # Group and calculate Mean or Median
                if agg_func == 'mean':
                    grouped = df_filtered.groupby('model_year')[
                        'price'].mean().reset_index()
                else:
                    grouped = df_filtered.groupby('model_year')[
                        'price'].median().reset_index()

                # Line Chart
                fig = px.line(grouped, x='model_year', y='price',
                              title=f'Average Price by Model Year ({agg_func.title()})',
                              markers=True,
                              labels={'model_year': 'Model Year',
                                      'price': 'Price'},
                              color_discrete_sequence=['grey'])

                st.plotly_chart(fig)

                # Insights
                max_price_row = grouped.loc[grouped['price'].idxmax()]
                min_price_row = grouped.loc[grouped['price'].idxmin()]
                st.markdown(f"""
                **Insights:**
                - **Highest price:** ${max_price_row['price']:,.0f} for {int(max_price_row['model_year'])} models
                - **Lowest price:** ${min_price_row['price']:,.0f} for {int(min_price_row['model_year'])} models
                - **Average of all prices:** ${grouped['price'].mean():,.0f}
                - **Years analyzed:** {grouped.shape[0]}
                """)

        if i == 4:
            # 5. Descriptive statistics for Condition Impact

            # Use session_state to maintain the button state
            if "show_grouped_bar" not in st.session_state:
                st.session_state.show_grouped_bar = False

            # Button for Grouped Bar Plot
            if st.button("Show Bar grouped Plot for Condition Impact"):
                st.session_state.show_grouped_bar = True

            if st.session_state.show_grouped_bar:
                st.write('Average price by vehicle model and condition comparison')

                # Interactive Selector
                group_col_option = st.selectbox(
                    "Group by:",
                    options=[('Model', 'model')],
                    format_func=lambda x: x[0],
                    key="agg_condition_impact"
                )
                group_col = group_col_option[1]

                # Slider for models
                top_n = st.slider("Top N Models:", min_value=5,
                                  max_value=30, value=10, step=1)

                # Data filtering
                df_filtered = df_vehicles.dropna(
                    subset=[group_col, 'condition', 'price'])

                # Mean calculation
                grouped = df_filtered.groupby([group_col, 'condition'])[
                    'price'].mean().reset_index()

                # Top N models filtering
                top_models = df_filtered[group_col].value_counts().nlargest(
                    top_n).index
                grouped = grouped[grouped[group_col].isin(top_models)]

                # Grouped Bar Plot
                fig = px.bar(grouped, x=group_col, y='price', color='condition', barmode='group',
                             title=f'Average Price by {group_col.capitalize()} and Condition',
                             labels={group_col: group_col.capitalize(
                             ), 'price': 'Average Price', 'condition': 'Condition'},
                             color_discrete_sequence=['#e0e0e0', '#d9d9d9', '#bfbfbf', '#a6a6a6', '#7f7f7f', "#555555"])

                st.plotly_chart(fig)

                # Insights
                most_listed_model = df_filtered[group_col].value_counts(
                ).idxmax()
                most_common_condition = df_filtered['condition'].value_counts(
                ).idxmax()
                top_price_row = grouped.loc[grouped['price'].idxmax()]

                st.markdown(f"""
                **Insights:**
                - **Most listed model:** {most_listed_model}
                - **Most common condition overall:** {most_common_condition}
                - **Highest avg. price:** {top_price_row[group_col]} ({top_price_row['condition']}) – ${top_price_row['price']:,.0f}
                - **Model-condition combinations analyzed:** {grouped.shape[0]}
                """)

        if i == 5:
            # 6. Descriptive statistics for Brand vs. Price

            # Use session_state to maintain the button state
            if 'brand' not in df_vehicles.columns:
                df_vehicles['brand'] = df_vehicles['model'].str.split().str[0]

            # Button for Bar Plot
            if "show_brand_price" not in st.session_state:
                st.session_state.show_brand_price = False

            # Botón para mostrar gráfico de marcas
            if st.button("Show Bar Plot for Brand vs. Price"):
                st.session_state.show_brand_price = True

            if st.session_state.show_brand_price:
                st.write('Average price by vehicle brand')

                # Slider top N brands selection
                top_n_brand = st.slider("Top N Brands:", min_value=5,
                                        max_value=30, value=10, step=1, key="agg_brand_price")

                # Data filtering
                df_clean = df_vehicles.dropna(subset=['brand', 'price'])

                # Get top N brands
                top_brands = df_clean['brand'].value_counts().nlargest(
                    top_n_brand).index
                filtered = df_clean[df_clean['brand'].isin(top_brands)]

                # Calculate average price by brand
                avg_price = filtered.groupby(
                    'brand')['price'].mean().reset_index()

                # Bar Plot
                fig = px.bar(avg_price, x='brand', y='price',
                             title=f'Average Price by Brand (Top {top_n_brand})',
                             labels={'brand': 'Brand',
                                     'price': 'Average Price'},
                             color_discrete_sequence=['#808080'])
                fig.update_layout(xaxis={'categoryorder': 'total descending'})

                st.plotly_chart(fig)

                # Insights
                most_exp_brand = avg_price.loc[avg_price['price'].idxmax()]
                least_exp_brand = avg_price.loc[avg_price['price'].idxmin()]
                most_listed_brand = filtered['brand'].value_counts().idxmax()

                st.markdown(f"""
                **Insights:**
                - **Most expensive brand (avg):** {most_exp_brand['brand']} – ${most_exp_brand['price']:,.0f}
                - **Least expensive brand (avg):** {least_exp_brand['brand']} – ${least_exp_brand['price']:,.0f}
                - **Most listed brand:** {most_listed_brand}
                - **Average price across these brands:** ${avg_price['price'].mean():,.0f}
                """)

        if i == 6:
            # 7. Descriptive statistics for Mileage vs. Price (Depreciation)

            # Use session_state to maintain the button state
            if "show_heatmap" not in st.session_state:
                st.session_state.show_heatmap = False

            # Button for Heatmap Plot
            if st.button("Show Heatmap Plot for Mileage vs. Price (Depreciation)"):
                st.session_state.show_heatmap = True

            if st.session_state.show_heatmap:
                st.write(
                    'Price and mileage depreciation, filtered by vehicle condition')

                # Interactive Dropdown
                conditions = ['All'] + \
                    sorted(df_vehicles['condition'].dropna().unique().tolist())
                selected_condition = st.selectbox(
                    "Condition:", options=conditions, key="agg_milage_price")

                # Data Filtering
                if selected_condition == 'All':
                    df_filtered = df_vehicles[['odometer', 'price']].dropna()
                else:
                    df_filtered = df_vehicles[df_vehicles['condition'] ==
                                              selected_condition][['odometer', 'price']].dropna()

                # Drop outliers (top 1%)
                df_filtered = df_filtered[df_filtered['price']
                                          < df_filtered['price'].quantile(0.99)]
                df_filtered = df_filtered[df_filtered['odometer']
                                          < df_filtered['odometer'].quantile(0.99)]

                # Heatmap Plot
                fig = px.density_heatmap(df_filtered, x='odometer', y='price',
                                         nbinsx=50, nbinsy=50,
                                         histfunc='count',
                                         title=f'Price vs. Mileage Heatmap (Condition: {selected_condition})',
                                         labels={
                                             'odometer': 'Mileage (Odometer)', 'price': 'Price'},
                                         color_continuous_scale='Greys')

                st.plotly_chart(fig)

                # Insights
                st.markdown(f"""
                **Insights:**
                - **Condition filter:** {selected_condition}
                - **Mileage range:** {df_filtered['odometer'].min():,.0f} – {df_filtered['odometer'].max():,.0f} miles
                - **Price range:** ${df_filtered['price'].min():,.0f} – ${df_filtered['price'].max():,.0f}
                - **Total listings analyzed:** {df_filtered.shape[0]}
                """)

        if i == 7:
            # 8. Descriptive statistics for Top Models Sales

            # Use session_state to maintain the button state
            if "show_pie_chart" not in st.session_state:
                st.session_state.show_pie_chart = False

            # Button for Bar chart
            if st.button("Show Pie Chart for Top Models Sales"):
                st.session_state.show_pie_chart = True

            if st.session_state.show_pie_chart:
                st.write('Proportion of Top model or brand Sales')

                # Interactive Dropdown
                group_option = st.selectbox(
                    "Group by:",
                    options=[('Model', 'model'), ('Brand', 'brand')],
                    format_func=lambda x: x[0],
                    key="agg_top_models_sales"
                )
                group_by = group_option[1]

                # Slider for top N
                top_n = st.slider("Top N:", min_value=5,
                                  max_value=30, value=10, step=1)

                # Data Filtering
                df_filtered = df_vehicles.dropna(subset=[group_by])
                top_counts = df_filtered[group_by].value_counts().nlargest(
                    top_n).reset_index()
                top_counts.columns = [group_by, 'count']
                top_counts = top_counts.sort_values('count')

                # Pie Chart
                fig = px.pie(top_counts, names=group_by, values='count',
                             title=f'Share of Top {top_n} {group_by.capitalize()}s Listed',
                             color_discrete_sequence=px.colors.sequential.Greys[3:])

                st.plotly_chart(fig)

                # Insights
                top_entry = top_counts.iloc[-1]
                total = top_counts['count'].sum()
                percent = (top_entry['count'] / total) * 100

                st.markdown(f"""
                **Insights:**
                - **Most listed {group_by}:** {top_entry[group_by]} with {top_entry['count']} listings
                - **Share in top {top_n}:** {percent:.1f}%
                - **Total listings analyzed:** {total}
                """)

        if i == 8:
            # 9. Descriptive statistics for Transmission Type Analysis

            # Use session_state to maintain the button state
            if "show_transmission_bar" not in st.session_state:
                st.session_state.show_transmission_bar = False

            # Button for Horizontal Bar Plot
            if st.button("Show Horizontal Bar Plot for Transmission Type Analysis"):
                st.session_state.show_transmission_bar = True

            if st.session_state.show_transmission_bar:
                st.write(
                    'Comparación del precio promedio o mediano según el tipo de transmisión')

                # Interactive Dropdown
                agg_option = st.selectbox(
                    "Aggregate:",
                    options=[('Mean', 'mean'), ('Median', 'median')],
                    format_func=lambda x: x[0],
                    key="agg_transmission_type"
                )
                agg_func = agg_option[1]

                # Data Filtering
                df_trans = df_vehicles.dropna(subset=['transmission', 'price'])

                # Group and price calculation
                if agg_func == 'mean':
                    grouped = df_trans.groupby('transmission')[
                        'price'].mean().reset_index()
                else:
                    grouped = df_trans.groupby('transmission')[
                        'price'].median().reset_index()

                grouped = grouped.sort_values('price', ascending=True)

                # Horizontal Bar Plot
                fig = px.bar(grouped, y='transmission', x='price',
                             orientation='h',
                             title=f'{agg_func.title()} Price by Transmission Type',
                             labels={'transmission': 'Transmission',
                                     'price': f'{agg_func.title()} Price'},
                             color_discrete_sequence=['#808080'])

                fig.update_layout(yaxis=dict(categoryorder='total ascending'))

                st.plotly_chart(fig)

                # Insights
                highest = grouped.iloc[-1]
                lowest = grouped.iloc[0]

                st.markdown(f"""
                **Insights:**
                - **Most expensive (avg):** {highest['transmission']} – ${highest['price']:,.0f}
                - **Least expensive (avg):** {lowest['transmission']} – ${lowest['price']:,.0f}
                - **Average across types:** ${grouped['price'].mean():,.0f}
                """)

        if i == 9:
            # 10. Descriptive statistics for Fuel Type Preference

            # Use session_state to maintain the button state
            if "show_fuel_box" not in st.session_state:
                st.session_state.show_fuel_box = False

            # Button for Box Plot
            if st.button("Show Box Plot for Fuel Type Preference"):
                st.session_state.show_fuel_box = True

            if st.session_state.show_fuel_box:
                st.write('Price distribution by fuel type')

                # Multiselect for fuel types
                fuel_options = sorted(
                    df_vehicles['fuel'].dropna().unique().tolist())
                selected_fuels = st.multiselect(
                    "Fuel Types:", options=fuel_options,
                    default=['gas', 'diesel', 'electric', 'hybrid'], key="agg_fuel_type"
                )

                if selected_fuels:
                    # Data Filtering
                    df_filtered = df_vehicles[df_vehicles['fuel'].isin(
                        selected_fuels)].dropna(subset=['price', 'fuel'])

                    # Drop outliers
                    df_filtered = df_filtered[df_filtered['price']
                                              < df_filtered['price'].quantile(0.99)]

                    # Box Plot
                    fig = px.box(df_filtered, x='fuel', y='price',
                                 title='Price Distribution by Fuel Type',
                                 labels={'fuel': 'Fuel Type',
                                         'price': 'Price'},
                                 color='fuel', color_discrete_sequence=px.colors.sequential.Blackbody)

                    fig.update_layout(xaxis_title='Fuel Type',
                                      yaxis_title='Price')

                    st.plotly_chart(fig)

                    # Insights
                    avg_prices = df_filtered.groupby('fuel')['price'].mean()
                    most_exp = avg_prices.idxmax()
                    least_exp = avg_prices.idxmin()
                    diff = avg_prices.max() - avg_prices.min()

                    st.markdown(f"""
                    **Insights:**
                    - **Most expensive (avg):** {most_exp} – ${avg_prices.max():,.0f}
                    - **Least expensive (avg):** {least_exp} – ${avg_prices.min():,.0f}
                    - **Difference between highest and lowest:** ${diff:,.0f}
                    - **Listings analyzed:** {df_filtered.shape[0]}
                    """)

                else:
                    st.warning("Please select at least one fuel type.")

        if i == 10:
            # 11. Descriptive statistics for Correlation Analysis of Vehicle Features

            # Use session_state to maintain the button state
            df_corr = df_vehicles.copy()

            # Coding Maps
            condition_map = {'salvage': 1, 'fair': 2, 'good': 3,
                             'excellent': 4, 'like new': 5, 'new': 6}
            fuel_map = {'gas': 1, 'diesel': 2,
                        'hybrid': 3, 'electric': 4, 'other': 0}
            transmission_map = {'manual': 0, 'automatic': 1, 'other': 0.5}

            # Applying codes
            df_corr['condition_score'] = df_corr['condition'].map(
                condition_map)
            df_corr['fuel_score'] = df_corr['fuel'].map(fuel_map)
            df_corr['transmission_score'] = df_corr['transmission'].map(
                transmission_map)

            # Dictionary
            all_vars = {
                'Price': 'price',
                'Odometer': 'odometer',
                'Model Year': 'model_year',
                'Days Listed': 'days_listed',
                'Is 4WD': 'is_4wd',
                'Condition Score': 'condition_score',
                'Fuel Score': 'fuel_score',
                'Transmission Score': 'transmission_score'
            }

            # Use session_state to maintain the button state
            if "show_scatter_matrix" not in st.session_state:
                st.session_state.show_scatter_matrix = False

            # Button for Scatter Matrix
            if st.button("Display scatter matrix of selected variables"):
                st.session_state.show_scatter_matrix = True

            if st.session_state.show_scatter_matrix:
                st.write('Dispersion matrix between selected variables')

                # Multiselect of variables
                selected_labels = st.multiselect(
                    "Variables:",
                    options=list(all_vars.keys()),
                    default=['Price', 'Odometer',
                             'Model Year', 'Condition Score'],
                    key="agg_correlation"
                )

                if len(selected_labels) >= 2:
                    selected_cols = [all_vars[label]
                                     for label in selected_labels]
                    df_selected = df_corr[selected_cols].dropna()

                    # Scatter Matrix Plot
                    fig = px.scatter_matrix(df_selected,
                                            dimensions=selected_cols,
                                            title="Scatter Matrix of Selected Features",
                                            color_discrete_sequence=['grey'])

                    fig.update_traces(diagonal_visible=True,
                                      showupperhalf=False)

                    st.plotly_chart(fig)

                    # Insights
                    corr_matrix = df_selected.corr()
                    corr_unstacked = corr_matrix.where(
                        ~np.eye(corr_matrix.shape[0], dtype=bool)).stack()
                    max_corr = corr_unstacked.abs().idxmax()
                    min_corr = corr_unstacked.abs().idxmin()

                    st.markdown(f"""
                    **Insights:**
                    - **Number of variables selected:** {len(selected_cols)}
                    - **Highest correlation:** {max_corr[0]} and {max_corr[1]} - {corr_matrix.loc[max_corr[0], max_corr[1]]:.2f}
                    - **Lowest correlation:** {min_corr[0]} and {min_corr[1]} - {corr_matrix.loc[min_corr[0], min_corr[1]]:.2f}
                    """)

                else:
                    st.warning(
                        "Select at least two variables to generate the matrix.")
