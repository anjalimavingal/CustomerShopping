import streamlit as st
from predict import predict
st.title("ML Classifier")
age = st.number_input("Age")
monthly_income = st.number_input("Monthly Income")
daily_internet_hours = st.number_input("Daily Internet Hours")
smartphone_usage_years = st.number_input("Years of smartphone usage")
social_media_hours = st.number_input("Daily social media usage")
online_payment_trust_score = st.number_input("Trust in digital payments")
tech_savvy_score = st.number_input("Comfort with technology")
monthly_online_orders = st.number_input("Number of online orders per month")
monthly_store_visits = st.number_input("Store visits per month")
avg_online_spend = st.number_input("Average online purchase value")
avg_store_spend = st.number_input("Average in-store purchase value")
discount_sensitivity = st.number_input("Response to discounts")
return_frequency = st.number_input("Online product return frequency")
avg_delivery_days = st.number_input("Typical delivery time")
delivery_fee_sensitivity = st.number_input("Sensitivity to delivery charges")
free_return_importance = st.number_input("Importance of free returns")
product_availability_online = st.number_input("Perceived online product availability")
impulse_buying_score = st.number_input("Likelihood of impulse purchases")
need_touch_feel_score = st.number_input("Preference to see/touch products before buying")
brand_loyalty_score = st.number_input("Brand loyalty level") 
environmental_awareness = st.number_input("Eco-consciousness level") 
time_pressure_level = st.number_input("Perceived lack of time") 
gender = st.text_input("Gender of the customer")
city_tier = st.text_input("city classification")
input = [age, monthly_income, daily_internet_hours, smartphone_usage_years, social_media_hours, online_payment_trust_score,
        tech_savvy_score, monthly_online_orders, monthly_store_visits, avg_online_spend, avg_store_spend, discount_sensitivity,
        return_frequency, avg_delivery_days, delivery_fee_sensitivity, free_return_importance, product_availability_online,
        impulse_buying_score, need_touch_feel_score, brand_loyalty_score, environmental_awareness, time_pressure_level, gender, city_tier ]
if st.button("Predict"):
    result = predict(input)
    st.write("Prediction:", result)