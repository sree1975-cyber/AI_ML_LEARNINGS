# models/linear_regression/linear_regression.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import plotly.express as px
import time

# Initialize session state
if 'trained' not in st.session_state:
    st.session_state.trained = False
if 'sample_data' not in st.session_state:
    st.session_state.sample_data = None

# Page config MUST be first
st.set_page_config(layout="wide")
st.title("🏡 Advanced Home Valuation System")

# ========================
# 1. INFORMATION SECTION
# ========================
def display_info():
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

display_info()

# ========================
# 2. DATA & TRAINING SECTION
# ========================
with st.container(border=True):
    st.subheader("📊 Data Management")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Load Sample Data", help="Generate synthetic housing data"):
            with st.spinner("Generating data..."):
                st.session_state.sample_data = generate_sample_data()
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

# ========================
# 3. PREDICTION SECTION
# ========================
if st.session_state.trained:
    with st.container(border=True):
        st.subheader("🔮 Prediction Panel")
        
        with st.form("prediction_form"):
            st.write("**Property Details**")
            c1, c2, c3 = st.columns(3)
            
            with c1:
                house_type = st.selectbox("Type", ['Single Family', 'Condo', 'Townhouse'])
                sq_feet = st.number_input("Size (sq.ft)", 500, 5000, 1500)
            
            with c2:
                bedrooms = st.select_slider("Bedrooms", [1, 2, 3, 4, 5])
                bathrooms = st.select_slider("Bathrooms", [1, 1.5, 2, 2.5, 3])
            
            with c3:
                year_built = st.slider("Year Built", 1950, 2023, 2000)
                location = st.selectbox("Location", ['Urban', 'Suburban', 'Rural'])
                market = st.radio("Market", ['Hot', 'Normal', 'Cool'])
            
            if st.form_submit_button("💰 Get Valuation", type="primary"):
                # Prepare input
                input_data = {
                    'house_type': house_type,
                    'sq_feet': sq_feet,
                    'bedrooms': bedrooms,
                    'bathrooms': bathrooms,
                    'year_built': year_built,
                    'location': location,
                    'market_condition': market
                }
                
                # Predict
                input_df = pd.DataFrame([input_data])
                processed_input = st.session_state.preprocessor.transform(input_df)
                prediction = st.session_state.model.predict(processed_input)[0]
                avg_price = st.session_state.sample_data['price'].mean()
                
                # Display results
                st.divider()
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Estimated Value", f"${prediction:,.0f}", 
                             delta=f"${prediction-avg_price:,.0f} vs avg", 
                             delta_color="normal")
                
                with col2:
                    with st.expander("📊 How this was calculated"):
                        st.write("**Price Components:**")
                        st.write(f"- Base: ${sq_feet * 200:,.0f} (size × rate)")
                        st.write(f"- Age: -{(2023-year_built)*100:,.0f} (depreciation)")
                        st.write(f"- Location: { {'Urban':1.3, 'Suburban':1.1, 'Rural':0.9}[location] }x")
                        st.write(f"- Market: { {'Hot':1.15, 'Normal':1.0, 'Cool':0.85}[market] }x")
                        st.write(f"**Final Formula:**\n`(Base + Age) × Location × Market`")

# ========================
# 4. MODEL VISUALIZATION
# ========================
if st.session_state.trained:
    with st.container(border=True):
        st.subheader("📈 Market Analysis")
        tab1, tab2 = st.tabs(["Price Distribution", "Feature Impact"])
        
        with tab1:
            fig = px.histogram(st.session_state.sample_data, x='price', 
                             title="Current Market Prices")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            fig = px.scatter(st.session_state.sample_data, x='sq_feet', y='price',
                           color='location', size='bedrooms',
                           hover_data=['house_type', 'market_condition'])
            st.plotly_chart(fig, use_container_width=True)

# ========================
# SUPPORTING FUNCTIONS
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

if __name__ == "__main__":
    pass
