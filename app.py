import streamlit as st
import pandas as pd
import requests
import json
import plotly.express as px

# ----------------------------
# Set Streamlit Page Configuration
# ----------------------------
st.set_page_config(page_title="AI Chat with Visuals", layout="wide")

# ----------------------------
# Configuration and Data Load
# ----------------------------

# DeepSeek API Key (store securely)
DEEPSEEK_API_KEY = "sk-65f90630f4954f0baa416fe1ec29fc83"

# Load the new dataset (data remains on backend)
DATA_PATH = "data/Sajjad_data.xlsx"  # Update with your dataset file path
df = pd.read_excel(DATA_PATH)

# Remove duplicate columns (if any) and strip extra spaces from column names
df = df.loc[:, ~df.columns.duplicated()]
df.columns = df.columns.str.strip()

# ----------------------------
# Sidebar: Pricing Plan and Filters
# ----------------------------
st.sidebar.header("User Settings & Filters")

# Pricing plan selection via buttons for easier checkout
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("Free Plan"):
        pricing_plan = "Free"
    else:
        pricing_plan = None
with col2:
    if st.button("Pricing 1"):
        pricing_plan = "Pricing 1"
    else:
        pricing_plan = None
with col3:
    if st.button("Pricing 2"):
        pricing_plan = "Pricing 2"
    else:
        pricing_plan = None

# Token optimization
pricing_cost = {"Free": 10, "Pricing 1": 20, "Pricing 2": 30, "Pricing 3": 40}
monthly_cap = {"Free": 50, "Pricing 1": 200, "Pricing 2": 300, "Pricing 3": 500}

# Stripe integration for checkout
import stripe
stripe.api_key = "your-stripe-secret-key"  # Replace with your Stripe secret key

def create_checkout_session(pricing_plan):
    try:
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price_data': {
                    'currency': 'usd',
                    'product_data': {
                        'name': pricing_plan,
                    },
                    'unit_amount': pricing_cost[pricing_plan] * 100,  # in cents
                },
                'quantity': 1,
            }],
            mode='payment',
            success_url=f"{st.get_url()}/success",
            cancel_url=f"{st.get_url()}/cancel",
        )
        return checkout_session
    except Exception as e:
        st.error(f"Error creating checkout session: {e}")
        return None

if pricing_plan:
    st.write(f"Selected Pricing Plan: {pricing_plan}")
    if st.button("Proceed to Payment"):
        session = create_checkout_session(pricing_plan)
        if session:
            st.write(f"Redirecting to Stripe checkout for the {pricing_plan} plan...")
            st.markdown(f'<a href="{session.url}" target="_blank">Click here to proceed to payment</a>', unsafe_allow_html=True)
else:
    st.write("Please select a pricing plan to continue.")

# ----------------------------
# Data Filtering (Optional)
# ----------------------------
filtered_df = df.copy()
# Add filters here if needed...

# ----------------------------
# Token/Spending Control Placeholder
# ----------------------------
# Implement token tracking logic if required

# ----------------------------
# Functions for AI Responses & Visualizations (Stubs)
# ----------------------------

def chat_with_data(user_question, data_context):
    """Simulated function to get AI response (Replace with real API call)."""
    # Example: You can integrate with OpenAI or DeepSeek API here
    return f"Simulated response for: {user_question}"

def get_ai_plot_instructions(data_context, question):
    """Simulated function to return plot instructions (Replace with real logic)."""
    # Example: Detect user intent and return plotting instructions
    return {
        "plots": [
            {
                "type": "bar",
                "x": "Column1",
                "y": "Column2",
                "title": "Sample Bar Chart"
            }
        ]
    }

def generate_ai_plots(df, plot_instructions):
    """Generate plots based on AI plot instructions."""
    for plot in plot_instructions["plots"]:
        if plot["type"] == "bar":
            fig = px.bar(df, x=plot["x"], y=plot["y"], title=plot["title"])
            st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Main Interaction
# ----------------------------

st.title("AI-Powered Dataset Insights")

user_question = st.text_input("Ask a question about the data:")
if user_question:
    st.write(f"Question: {user_question}")
    
    # ✅ Use the entire dataset for context
    data_context = df.to_json(orient="split")
    
    ai_response = chat_with_data(user_question, data_context)
    st.write(f"AI Response: {ai_response}")
    
    # Get AI plot instructions and generate plots
    ai_plots = get_ai_plot_instructions(data_context, user_question)
    if ai_plots.get("plots"):
        generate_ai_plots(filtered_df, ai_plots)
    else:
        st.write("No plots available for this question.")
