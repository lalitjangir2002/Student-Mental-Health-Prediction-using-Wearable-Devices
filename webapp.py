import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st
from pymongo import MongoClient
from bson.objectid import ObjectId
import re

# Load the trained model
model = joblib.load('mental_health_predictor_ensemble.joblib')

# MongoDB connection
def get_database_connection():
    """Connect to MongoDB."""
    try:
        client = MongoClient('')
        db = client['mentalHealthDB']
        collection = db['dashboards']
        docs = list(collection.find({'_id' : ObjectId("680893c53f20e5bc586961ce")}))
        # st.write(docs)
        # for doc in docs:
        #     st.write(doc.get("_id"))
        
        st.write("✅ Successfully connected to MongoDB")
        return db
    except Exception as e:
        st.error(f"❌ Failed to connect to MongoDB: {str(e)}")
        return None

def display_collection_data(db, collection_name):
    """Display MongoDB collection data in a table format."""
    try:
        st.write(f"📊 Fetching data from {collection_name} collection...")
        documents = list(db[collection_name].find())

        if not documents:
            st.write(f"⚠️ No data found in {collection_name} collection")
            return

        # Convert ObjectIds to string
        # for doc in documents:
        #     doc['_id'] = str(doc['_id'])

        df = pd.DataFrame(documents)
        st.write(f"✅ Found {len(df)} records")

        st.subheader(f"{collection_name.replace('_', ' ').title()} Data")
        st.dataframe(df)

    except Exception as e:
        st.error(f"❌ Error displaying {collection_name} data: {str(e)}")

def fetch_user_data(db, user_id, date):
    """Fetch user data from MongoDB for a specific date."""
    try:
        st.write(f"🔍 Fetching data for user {user_id} on {date}")
        
        # Convert date string to datetime objects
        start_date = datetime.strptime(date, "%Y-%m-%d")
        end_date = start_date + timedelta(days=1)
        
        # Query data
        query = {
            '_id': user_id,
            'createdAt': {
                '$gte': start_date,
                '$lt': end_date
            }
        }
        
        health_data = list(db.dashboards.find(query).sort('createdAt', 1))
        
        if not health_data:
            st.write("⚠️ No data found for the specified user and date")
            return None
            
        st.write(f"✅ Found {len(health_data)} records")
        
        # Convert to DataFrame
        df = pd.DataFrame(health_data)
        st.write("✅ Successfully converted data to DataFrame")
        
        # Preprocess and make prediction
        prediction = make_prediction(df)
        
        if prediction is not None:
            st.subheader("Prediction Results")
            st.write(f"🎯 Predicted Mental Health Score: {prediction:.2f}")
        else:
            st.error("❌ Prediction failed")
        
        return df
        
    except Exception as e:
        st.error(f"❌ Error fetching data: {str(e)}")
        return None

def preprocess_user_data(user_data):
    """Renames columns to match the trained model's expected input and preprocesses the data."""
    # st.write(user_data)
    column_mapping = {
        'heartRateAvg': 'Heart_Rate_BPM',
        'oxygenAvg': 'Oxygen_Saturation',
        'temperature': 'Body_Temperature_Celsius',
        'totalSteps': 'Physical_Activity_Steps',
        'sleepDuration': 'Sleep_Duration_Hours'
    }

    # Rename columns to match model features
    user_data = user_data.rename(columns=column_mapping)
    # st.write(user_data)
    
    # Ensure that only the relevant columns are included
    features = ['Heart_Rate_BPM', 'Sleep_Duration_Hours', 'Physical_Activity_Steps', 
                'Oxygen_Saturation', 'Body_Temperature_Celsius']
    
    user_data = user_data[features]
    # st.write(user_data)
    
    user_data['Sleep_Duration_Hours'] = user_data['Sleep_Duration_Hours'].apply(convert_sleep_duration_to_float)
    
    # st.write(user_data)
    
    # Optionally, apply any preprocessing steps like scaling if required
    # For example, if your model was trained on scaled data, apply scaling here
    # scaler = StandardScaler()
    # user_data[features] = scaler.transform(user_data[features])
    
    return user_data

def convert_sleep_duration_to_float(duration_str):
    """Convert time duration string (e.g., '5 hr 12 min') to float (hours)."""
    try:
        # Regular expression to extract hours and minutes
        hours, minutes = 0, 0
        if 'hr' in duration_str and 'min' in duration_str:
            match = re.match(r"(\d+)\s*hr\s*(\d+)\s*min", duration_str)
            if match:
                hours = int(match.group(1))
                minutes = int(match.group(2))
        elif 'hr' in duration_str:
            hours = int(re.match(r"(\d+)\s*hr", duration_str).group(1))
        elif 'min' in duration_str:
            minutes = int(re.match(r"(\d+)\s*min", duration_str).group(1))

        # Convert minutes to a fraction of an hour
        return hours + minutes / 60
    except Exception as e:
        st.error(f"❌ Error converting sleep duration: {str(e)}")
        return None

def make_prediction(user_data):
    """Make prediction on the user data."""
    try:
        # Preprocess the user data
        preprocessed_data = preprocess_user_data(user_data)
        
        # Make prediction using the VotingClassifier
        prediction = model.predict(preprocessed_data)[0]
        
        # Return the prediction
        return prediction
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
        return None

def main():
    st.title("Student Mental Health Prediction")
    
    # Connect to MongoDB
    db = get_database_connection()
    if db is None:
        return
    
    # Create tabs for different sections
    tab1, tab2 = st.tabs(["Database View", "Make Prediction"])
    
    with tab1:
        st.header("Database Collections")
        collections = db.list_collection_names()
        selected_collection = st.selectbox("Select Collection", collections)
        
        if selected_collection:
            display_collection_data(db, selected_collection)
    
    with tab2:
        st.header("Make Prediction")
        user_id = st.text_input("Enter User ID")
        date = st.date_input("Select date", datetime.now())
        
        if st.button("Analyze Data"):
            if not user_id:
                st.warning("⚠️ Please enter a User ID")
                return
                
            with st.spinner("Analyzing data..."):
                # Fetch user data
                user_data = fetch_user_data(db, ObjectId(user_id), date.strftime("%Y-%m-%d"))
                
                if user_data is not None:
                    try:
                        # Debug: Show raw data
                        st.subheader("Raw Data")
                        st.dataframe(user_data)
                        
                        # Make prediction
                        st.write("🤖 Making prediction...")
                        prediction = make_prediction(user_data)
                        
                        # Display results
                        st.subheader("Prediction Results")
                        st.write(f"🎯 Predicted Mental Health Score: {prediction:.2f}")
                        
                    except Exception as e:
                        st.error(f"❌ Error during prediction: {str(e)}")
                else:
                    st.error("❌ No data available for prediction")

if __name__ == "__main__":
    main()
