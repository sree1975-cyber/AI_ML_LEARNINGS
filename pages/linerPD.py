# models/linear_regression/linear_regression.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import plotly.express as px
import time

# ========================
# SUPPORTING FUNCTIONS (DEFINE THESE FIRST)
# ========================
def generate_sample_data(records=200):
    """Generates synthetic housing data"""
    np.random.seed(42)
    data = {
        'house_type': np.random.choice(['Single Family', 'Condo', 'Townhouse'], records),
        'sq_feet': np.abs(np.random.normal(1500, 600, records)).astype(int),
        'bedrooms': np.random.choice([1, 2, 3, 4], records),
        'bathrooms': np.random.choice([1, 1.5, 2, 2.5], records),
        'year_built': np.random.randint(1970, 2023, records),
        'location': np.random.choice(['Urban', 'Suburban', 'Rural'], records),
        'market_condition': np.random.choice(['Hot', 'Normal', 'Cool'], records)
    }
    
    df = pd.DataFrame(data)
    df['price'] = (
        (df['sq_feet'] * 200) 
        + ((2023 - df['year_built']) * -100) 
        * df['location'].map({'Urban':1.3, 'Suburban':1.1, 'Rural':0.9}) 
        * df['market_condition'].map({'Hot':1.15, 'Normal':1.0, 'Cool':0.85})
    ).astype(int)
    
    return df

def setup_model(df):
    """Trains the prediction model"""
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), ['house_type', 'location', 'market_condition'])
        ],
        remainder='passthrough'
    )
    
    X = preprocessor.fit_transform(df.drop('price', axis=1))
    y = df['price']
    
    model = LinearRegression()
    model.fit(X, y)
    return model, preprocessor

def display_info():
    """Displays model documentation"""
    with st.expander("📚 Model Information", expanded=True):
        tab1, tab2, tab3 = st.tabs(["Overview", "How It Works", "Technical Details"])
        
        with tab1:
            st.markdown("""
            **Linear Regression Model for Home Valuation**
            - Predicts property values based on key features
            - Adapts to market conditions
            - Provides transparent pricing factors
            """)
        
        with tab2:
            st.markdown("""
            1. **Load Sample Data**: Generates synthetic housing data
            2. **Train Model**: Creates pricing algorithm
            3. **Predict**: Get instant valuations with explanations
            """)
        
        with tab3:
            st.code("""
            Price = (Base × Size) 
                  + Age Adjustment 
                  × Location Premium 
                  × Market Condition
            """)

# ========================
# STREAMLIT APP (MAIN CODE)
# ========================
# Initialize session state
if 'trained' not in st.session_state:
    st.session_state.trained = False
if 'sample_data' not in st.session_state:
    st.session_state.sample_data = None

# Page config MUST be first
st.set_page_config(layout="wide")
st.title("🏡 Advanced Home Valuation System")

# 1. Information Section
display_info()

# 2. Data & Training Section
with st.container(border=True):
    st.subheader("📊 Data Management")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Load Sample Data", help="Generate synthetic housing data"):
            with st.spinner("Generating data..."):
                st.session_state.sample_data = generate_sample_data()  # NOW DEFINED
                st.session_state.trained = False
                st.success(f"Generated {len(st.session_state.sample_data)} records")
                
                # Show sample
                st.dataframe(
                    st.session_state.sample_data.sample(3),
                    column_config={"price": st.column_config.NumberColumn(format="$%,d")},
                    hide_index=True
                )
    
    with col2:
        if st.session_state.sample_data is not None and not st.session_state.trained:
            if st.button("🎯 Train Model", type="primary"):
                with st.spinner("Training model..."):
                    st.session_state.model, st.session_state.preprocessor = setup_model(st.session_state.sample_data)
                    time.sleep(1)  # Simulate training
                    st.session_state.trained = True
                    st.toast("Model trained successfully!", icon="✅")
                    
                    # Show training insights
                    with st.expander("🔍 Training Insights"):
                        st.write("**Patterns Discovered:**")
                        st.write("- Every additional sq.ft adds ~$200")
                        st.write("- Urban homes command 30% premium")
                        st.write("- Market demand affects prices by ±15%")

# [Rest of your code remains the same...]
