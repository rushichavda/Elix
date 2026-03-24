import streamlit as st
import sqlite3
import json
import requests
from datetime import datetime, timedelta
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional
import re


# Configuration
import os
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
MODEL = "gpt-3.5-turbo"  # or "gpt-4" if you prefer

# 20 Most Popular Exercises with categories and muscle groups - REP-BASED ONLY
EXERCISES_DB = {
    "pushups": {"category": "strength", "muscle_groups": ["chest", "shoulders", "triceps"]},
    "squats": {"category": "strength", "muscle_groups": ["legs", "glutes"]},
    "pullups": {"category": "strength", "muscle_groups": ["back", "biceps"]},
    "deadlifts": {"category": "strength", "muscle_groups": ["back", "legs", "glutes"]},
    "bench press": {"category": "strength", "muscle_groups": ["chest", "shoulders", "triceps"]},
    "lunges": {"category": "strength", "muscle_groups": ["legs", "glutes"]},
    "burpees": {"category": "cardio", "muscle_groups": ["full body"]},
    "mountain climbers": {"category": "cardio", "muscle_groups": ["core", "shoulders"]},
    "jumping jacks": {"category": "cardio", "muscle_groups": ["full body"]},
    "bicep curls": {"category": "strength", "muscle_groups": ["biceps"]},
    "tricep dips": {"category": "strength", "muscle_groups": ["triceps"]},
    "shoulder press": {"category": "strength", "muscle_groups": ["shoulders"]},
    "crunches": {"category": "strength", "muscle_groups": ["core"]},
    "leg press": {"category": "strength", "muscle_groups": ["legs", "glutes"]},
    "lat pulldowns": {"category": "strength", "muscle_groups": ["back", "biceps"]},
    "hip thrusts": {"category": "strength", "muscle_groups": ["glutes"]},
    "russian twists": {"category": "strength", "muscle_groups": ["core"]},
    "sit ups": {"category": "strength", "muscle_groups": ["core"]},
    "leg raises": {"category": "strength", "muscle_groups": ["core"]},
    "dumbbell rows": {"category": "strength", "muscle_groups": ["back", "biceps"]}
}

class DatabaseManager:
    def __init__(self, db_path="fitness_tracker.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create exercises table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS exercises (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT DEFAULT 'default_user',
                date TEXT,
                exercise_name TEXT,
                sets INTEGER,
                reps INTEGER,
                category TEXT,
                muscle_groups TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create user_stats table for quick lookups
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT DEFAULT 'default_user',
                date TEXT,
                total_exercises INTEGER,
                total_sets INTEGER,
                total_reps INTEGER,
                categories_worked TEXT,
                muscle_groups_worked TEXT,
                UNIQUE(user_id, date)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_exercise(self, exercise_data: Dict):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO exercises (date, exercise_name, sets, reps, category, muscle_groups)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            exercise_data['date'],
            exercise_data['exercise_name'],
            exercise_data['sets'],
            exercise_data['reps'],
            exercise_data['category'],
            json.dumps(exercise_data['muscle_groups'])
        ))
        
        conn.commit()
        conn.close()
        
        # Update daily stats
        self.update_daily_stats(exercise_data['date'])
    
    def update_daily_stats(self, date: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get daily aggregated data - COUNT DISTINCT exercises, not total entries
        cursor.execute('''
            SELECT 
                COUNT(DISTINCT exercise_name) as unique_exercises,
                SUM(sets) as total_sets,
                SUM(sets * reps) as total_reps,
                GROUP_CONCAT(DISTINCT category) as categories
            FROM exercises 
            WHERE date = ?
        ''', (date,))
        
        result = cursor.fetchone()
        
        if result and result[0] > 0:
            # Get all muscle groups for the day
            cursor.execute('SELECT muscle_groups FROM exercises WHERE date = ?', (date,))
            muscle_groups_raw = cursor.fetchall()
            all_muscle_groups = set()
            for mg_json in muscle_groups_raw:
                mg_list = json.loads(mg_json[0])
                all_muscle_groups.update(mg_list)
            
            # Insert or replace daily stats
            cursor.execute('''
                INSERT OR REPLACE INTO user_stats 
                (user_id, date, total_exercises, total_sets, total_reps, categories_worked, muscle_groups_worked)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                'default_user',
                date,
                result[0],  # Now this is unique exercises count
                result[1] or 0,
                result[2] or 0,
                result[3] or '',
                json.dumps(list(all_muscle_groups))
            ))
        
        conn.commit()
        conn.close()
    
    def get_exercises_by_date_range(self, start_date: str, end_date: str) -> pd.DataFrame:
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query('''
            SELECT * FROM exercises 
            WHERE date BETWEEN ? AND ?
            ORDER BY date DESC, timestamp DESC
        ''', conn, params=(start_date, end_date))
        conn.close()
        return df
    
    def get_daily_stats(self, start_date: str, end_date: str) -> pd.DataFrame:
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query('''
            SELECT * FROM user_stats 
            WHERE date BETWEEN ? AND ?
            ORDER BY date DESC
        ''', conn, params=(start_date, end_date))
        conn.close()
        return df
    
    def delete_exercise(self, exercise_id: int):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get the date before deleting
        cursor.execute('SELECT date FROM exercises WHERE id = ?', (exercise_id,))
        result = cursor.fetchone()
        
        if result:
            date = result[0]
            cursor.execute('DELETE FROM exercises WHERE id = ?', (exercise_id,))
            conn.commit()
            conn.close()
            
            # Update daily stats
            self.update_daily_stats(date)
        else:
            conn.close()

class ExerciseParser:
    def __init__(self):
        self.exercises_list = list(EXERCISES_DB.keys())
    
    def call_llm(self, prompt: str) -> str:
        try:
            response = requests.post(
                url="https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                data=json.dumps({
                    "model": MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1
                })
            )
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                return f"Error: {response.status_code} - {response.text}"
        except Exception as e:
            return f"Error calling LLM: {str(e)}"
    
    def parse_exercise_input(self, user_input: str) -> List[Dict]:
        exercises_str = ", ".join(self.exercises_list)
        
        prompt = f"""
You are a fitness tracking assistant. Parse the user's exercise input and extract structured data.

Available exercises (only use these): {exercises_str}

User input: "{user_input}"

Instructions:
1. Identify which exercises from the available list the user mentioned (use closest match if needed)
2. Extract sets and reps for each exercise
3. If user mentions just reps without sets, assume 1 set
4. If user mentions just sets without reps, ask for clarification
5. Return ONLY a valid JSON array in this exact format:

[
    {{
        "exercise_name": "exact_name_from_available_list",
        "sets": number,
        "reps": number,
        "confidence": "high/medium/low"
    }}
]

If you cannot parse the input or exercise is not in the available list, return:
[{{"error": "explanation_of_issue"}}]

Examples:
- "I did 3 sets of 10 pushups" → [{{"exercise_name": "pushups", "sets": 3, "reps": 10, "confidence": "high"}}]
- "20 squats and 15 burpees" → [{{"exercise_name": "squats", "sets": 1, "reps": 20, "confidence": "high"}}, {{"exercise_name": "burpees", "sets": 1, "reps": 15, "confidence": "high"}}]
"""
        
        response = self.call_llm(prompt)
        print(f"LLM Response: {response}")  # Debugging line
        try:
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed_data = json.loads(json_str)
                return parsed_data
            else:
                return [{"error": "Could not extract valid JSON from LLM response"}]
        except json.JSONDecodeError:
            return [{"error": "Invalid JSON format in LLM response"}]

class FitnessTracker:
    def __init__(self):
        self.db = DatabaseManager()
        self.parser = ExerciseParser()
    
    def add_workout(self, user_input: str, date: str = None) -> str:
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        parsed_exercises = self.parser.parse_exercise_input(user_input)
        
        if not parsed_exercises:
            return "❌ Could not parse your input. Please try again with clearer exercise details."
        
        if "error" in parsed_exercises[0]:
            return f"❌ {parsed_exercises[0]['error']}"
        
        success_count = 0
        results = []
        
        for exercise in parsed_exercises:
            if "error" not in exercise:
                exercise_name = exercise['exercise_name']
                
                if exercise_name in EXERCISES_DB:
                    exercise_data = {
                        'date': date,
                        'exercise_name': exercise_name,
                        'sets': exercise['sets'],
                        'reps': exercise['reps'],
                        'category': EXERCISES_DB[exercise_name]['category'],
                        'muscle_groups': EXERCISES_DB[exercise_name]['muscle_groups']
                    }
                    
                    self.db.add_exercise(exercise_data)
                    results.append(f"✅ {exercise_name}: {exercise['sets']} sets × {exercise['reps']} reps")
                    success_count += 1
                else:
                    results.append(f"❌ '{exercise_name}' not found in exercise database")
        
        if success_count > 0:
            return f"🎉 Successfully logged {success_count} exercise(s):\n" + "\n".join(results)
        else:
            return "❌ No valid exercises were logged. Please check your input."

def create_dashboard():
    st.set_page_config(page_title="Fitness Tracker", page_icon="💪", layout="wide")
    
    # Initialize tracker
    if 'tracker' not in st.session_state:
        st.session_state.tracker = FitnessTracker()
    
    tracker = st.session_state.tracker
    
    st.title("💪 Fitness Tracker Chatbot")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    tab = st.sidebar.selectbox("Choose Section", ["Log Exercise", "Dashboard", "Manage Exercises"])
    
    if tab == "Log Exercise":
        st.header("🏋️ Log Your Exercise")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            user_input = st.text_input(
                "Tell me about your workout:",
                placeholder="e.g., I did 3 sets of 10 pushups and 20 squats",
                key="exercise_input"
            )
        
        with col2:
            date_input = st.date_input("Date", datetime.now())
        
        if st.button("Log Exercise", type="primary"):
            if user_input.strip():
                result = tracker.add_workout(user_input, date_input.strftime("%Y-%m-%d"))
                st.success(result) if "Successfully logged" in result else st.error(result)
                
                # Clear input after successful logging
                if "Successfully logged" in result:
                    st.rerun()
            else:
                st.warning("Please enter your exercise details!")
        
        # Show available exercises
        st.subheader("📋 Available Exercises")
        exercises_by_category = {}
        for exercise, data in EXERCISES_DB.items():
            category = data['category'].title()
            if category not in exercises_by_category:
                exercises_by_category[category] = []
            exercises_by_category[category].append(exercise.title())
        
        for category, exercises in exercises_by_category.items():
            st.write(f"**{category}:** {', '.join(exercises)}")
    
    elif tab == "Dashboard":
        st.header("📊 Your Fitness Dashboard")
        
        # Date range selector with better default
        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            start_date = st.date_input("From", datetime.now() - timedelta(days=30))
        with col2:
            end_date = st.date_input("To", datetime.now())
        with col3:
            if st.button("Last 7 Days"):
                start_date = datetime.now() - timedelta(days=7)
                end_date = datetime.now()
                st.rerun()
        
        # Get data
        daily_stats = tracker.db.get_daily_stats(
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        )
        
        exercises_df = tracker.db.get_exercises_by_date_range(
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d")
        )
        
        if daily_stats.empty:
            st.info("No workout data found for the selected date range. Start logging your rep-based exercises!")
            
            # Show example inputs
            st.subheader("💡 Example Inputs")
            examples = [
                "I did 3 sets of 10 pushups",
                "20 squats and 15 burpees",
                "3 sets of 8 pullups and 2 sets of 12 bicep curls",
                "100 jumping jacks, 50 crunches, 30 mountain climbers"
            ]
            for example in examples:
                st.code(example)
            return
        
        # Key metrics - meaningful ones only
        st.subheader("📈 Activity Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        total_days = len(daily_stats)
        total_exercises = daily_stats['total_exercises'].sum()
        total_days_range = (end_date - start_date).days + 1
        avg_exercises_per_day = total_exercises / total_days if total_days > 0 else 0
        
        # Calculate days since first workout for better context
        if not exercises_df.empty:
            first_workout_date = pd.to_datetime(exercises_df['date']).min().date()
            days_since_start = (end_date - first_workout_date).days + 1
            activity_rate_since_start = (total_days / days_since_start) * 100 if days_since_start > 0 else 0
        else:
            days_since_start = total_days_range
            activity_rate_since_start = 0
        
        # Calculate streaks
        def calculate_streaks(daily_stats_df):
            if daily_stats_df.empty:
                return 0, 0, 0
            
            # Sort by date
            df_sorted = daily_stats_df.sort_values('date')
            dates = pd.to_datetime(df_sorted['date']).dt.date.tolist()
            
            # Create a complete date range from first to last workout
            if dates:
                all_dates = [d.date() for d in pd.date_range(start=min(dates), end=max(dates), freq='D')]
                workout_dates_set = set(dates)
                
                # Calculate longest streak
                longest_streak = 0
                current_longest = 0
                
                for date in all_dates:
                    if date in workout_dates_set:
                        current_longest += 1
                        longest_streak = max(longest_streak, current_longest)
                    else:
                        current_longest = 0
                
                # Calculate current streak (from today backwards)
                current_streak = 0
                today = datetime.now().date()
                
                # Check if today or recent days have workouts
                check_date = today
                while check_date >= min(dates):
                    if check_date in workout_dates_set:
                        current_streak += 1
                        check_date -= timedelta(days=1)
                    else:
                        break
                
                # Calculate days since last workout
                days_since_last = (today - max(dates)).days if dates else 0
                
                return longest_streak, current_streak, days_since_last
            
            return 0, 0, 0
        
        longest_streak, current_streak, days_since_last = calculate_streaks(daily_stats)
        
        col1.metric("Active Days", f"{total_days}/{total_days_range}")
        col2.metric("Total Exercise Sessions", total_exercises)
        col3.metric("Avg Exercises/Day", f"{avg_exercises_per_day:.1f}")
        
        # Show contextual activity rate or streak info
        if days_since_start <= 7:
            # For new users (less than a week) - show current streak
            if current_streak > 0:
                col4.metric("Current Streak", f"{current_streak} days", "🔥 Keep it up!")
            else:
                col4.metric("Days Active", f"{total_days} days", "since you started")
        elif days_since_start <= 30:
            # For users with 1-4 weeks of data - show longest streak
            col4.metric("Longest Streak", f"{longest_streak} days", f"Current: {current_streak}" if current_streak > 0 else f"Last workout: {days_since_last} days ago")
        else:
            # For established users - show both streaks
            if current_streak > 0:
                col4.metric("Current Streak", f"{current_streak} days", f"Best: {longest_streak} days")
            else:
                col4.metric("Longest Streak", f"{longest_streak} days", f"Last workout: {days_since_last} days ago")
        
        # Add streak insights below metrics
        if longest_streak > 0 or current_streak > 0:
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if longest_streak > 0:
                    st.metric("🏆 Best Streak", f"{longest_streak} days")
            
            with col2:
                if current_streak > 0:
                    st.metric("🔥 Current Streak", f"{current_streak} days")
                else:
                    st.metric("⏱️ Days Since Last Workout", f"{days_since_last} days")
            
            with col3:
                # Streak motivation
                if current_streak >= longest_streak and current_streak > 0:
                    st.metric("🎯 Status", "New Record!", "🌟")
                elif current_streak > 0:
                    remaining = longest_streak - current_streak + 1
                    st.metric("🎯 To Beat Record", f"{remaining} more days")
                else:
                    st.metric("🎯 Streak Goal", "Start today!")
        
        # Daily Activity Level (keeping this as it shows general activity)
        st.subheader("📊 Daily Activity Level")
        if not daily_stats.empty:
            # Create activity chart based on total reps (as activity indicator)
            # Convert date to datetime and format for cleaner display
            daily_stats_clean = daily_stats.copy()
            daily_stats_clean['date'] = pd.to_datetime(daily_stats_clean['date']).dt.date
            
            fig_activity = px.bar(daily_stats_clean, x='date', y='total_reps', 
                         title='Daily Activity Level (Total Reps as Activity Indicator)',
                         color='total_reps',
                         color_continuous_scale='viridis')
            fig_activity.update_layout(showlegend=False, height=400)
            fig_activity.update_xaxes(title="Date", type='category')
            fig_activity.update_yaxes(title="Activity Level (Total Reps)")
            st.plotly_chart(fig_activity, use_container_width=True)
        
        # Muscle Group Analysis - This is where the real insights are
        st.subheader("💪 Muscle Group Analysis")
        
        if not exercises_df.empty:
            # Calculate muscle group training frequency and balance
            muscle_group_data = {}
            muscle_group_by_date = {}
            
            for _, row in exercises_df.iterrows():
                date = row['date']
                muscle_groups = json.loads(row['muscle_groups'])
                
                for mg in muscle_groups:
                    # Overall frequency
                    muscle_group_data[mg] = muscle_group_data.get(mg, 0) + 1
                    
                    # By date for tracking consistency
                    if date not in muscle_group_by_date:
                        muscle_group_by_date[date] = set()
                    muscle_group_by_date[date].add(mg)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Muscle group training frequency
                if muscle_group_data:
                    mg_df = pd.DataFrame(list(muscle_group_data.items()), 
                                       columns=['Muscle Group', 'Training Sessions'])
                    mg_df = mg_df.sort_values('Training Sessions', ascending=True)
                    
                    fig_mg = px.bar(mg_df, x='Training Sessions', y='Muscle Group',
                                   title='Muscle Group Training Frequency',
                                   orientation='h',
                                   color='Training Sessions',
                                   color_continuous_scale='plasma')
                    fig_mg.update_layout(showlegend=False, height=400)
                    st.plotly_chart(fig_mg, use_container_width=True)
            
            with col2:
                # Muscle group balance analysis
                if muscle_group_data:
                    total_sessions = sum(muscle_group_data.values())
                    mg_balance = {mg: (count/total_sessions)*100 for mg, count in muscle_group_data.items()}
                    
                    balance_df = pd.DataFrame(list(mg_balance.items()), 
                                            columns=['Muscle Group', 'Percentage'])
                    
                    fig_balance = px.pie(balance_df, values='Percentage', names='Muscle Group',
                                       title='Training Balance Across Muscle Groups',
                                       color_discrete_sequence=px.colors.qualitative.Set3)
                    fig_balance.update_layout(height=400)
                    st.plotly_chart(fig_balance, use_container_width=True)
        
        # Exercise Category and Frequency Analysis  
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Exercise Categories")
            if not exercises_df.empty:
                category_counts = exercises_df['category'].value_counts()
                fig_cat = px.pie(values=category_counts.values, names=category_counts.index,
                               title='Workout Categories Distribution',
                               color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1'])
                fig_cat.update_layout(height=400)
                st.plotly_chart(fig_cat, use_container_width=True)
        
        with col2:
            st.subheader("🏆 Most Frequent Exercises")
            if not exercises_df.empty:
                exercise_counts = exercises_df['exercise_name'].value_counts().head(8)
                fig_ex = px.bar(x=exercise_counts.values, y=exercise_counts.index,
                               title='Top Exercises by Frequency',
                               orientation='h',
                               color=exercise_counts.values,
                               color_continuous_scale='viridis')
                fig_ex.update_layout(showlegend=False, height=400)
                fig_ex.update_yaxes(title='Exercise')
                fig_ex.update_xaxes(title='Times Performed')
                st.plotly_chart(fig_ex, use_container_width=True)
        
        # Muscle Group Training Consistency Over Time
        st.subheader("📅 Muscle Group Training Consistency")
        if not exercises_df.empty and muscle_group_by_date:
            # Create a heatmap showing which muscle groups were trained on which days
            # Clean dates to remove timestamps
            all_dates = sorted(muscle_group_by_date.keys())
            all_dates_clean = [pd.to_datetime(date).strftime('%Y-%m-%d') for date in all_dates]
            all_muscle_groups = sorted(set().union(*muscle_group_by_date.values()))
            
            # Create matrix for heatmap
            consistency_matrix = []
            for mg in all_muscle_groups:
                row = []
                for date in all_dates:
                    row.append(1 if mg in muscle_group_by_date.get(date, set()) else 0)
                consistency_matrix.append(row)
            
            if consistency_matrix:
                fig_consistency = px.imshow(
                    consistency_matrix,
                    x=all_dates_clean,  # Use cleaned dates
                    y=all_muscle_groups,
                    color_continuous_scale='RdYlGn',
                    title='Muscle Group Training Consistency Over Time',
                    labels={'x': 'Date', 'y': 'Muscle Group', 'color': 'Trained'}
                )
                fig_consistency.update_layout(
                    height=500,  # Increased from 400 to 600
                    width=None,  # Let it use full container width
                    margin=dict(l=100, r=50, t=50, b=100)  # Add margins for better spacing
                )
                fig_consistency.update_xaxes(tickangle=45, type='category')
                fig_consistency.update_yaxes(tickfont=dict(size=12))  # Make y-axis labels more readable
                st.plotly_chart(fig_consistency, use_container_width=True)
                
                # Add explanation
                st.info("💡 **Interpretation**: Green indicates the muscle group was trained that day. Use this to identify gaps in your training and ensure balanced muscle development.")
        
        # Training Insights and Recommendations
        st.subheader("🧠 Training Insights")
        if not exercises_df.empty and muscle_group_data:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🎯 Training Balance Analysis:**")
                
                # Find most and least trained muscle groups
                most_trained = max(muscle_group_data, key=muscle_group_data.get)
                least_trained = min(muscle_group_data, key=muscle_group_data.get)
                
                st.write(f"• Most trained: **{most_trained.title()}** ({muscle_group_data[most_trained]} sessions)")
                st.write(f"• Least trained: **{least_trained.title()}** ({muscle_group_data[least_trained]} sessions)")
                
                # Balance ratio
                balance_ratio = muscle_group_data[most_trained] / muscle_group_data[least_trained]
                if balance_ratio > 3:
                    st.warning(f"⚠️ Training imbalance detected! {most_trained.title()} is trained {balance_ratio:.1f}x more than {least_trained.title()}")
                else:
                    st.success("✅ Training appears relatively balanced across muscle groups")
            
            with col2:
                st.write("**📊 Activity Patterns:**")
                
                # Calculate average exercises per active day
                avg_per_day = total_exercises / total_days if total_days > 0 else 0
                st.write(f"• Average exercises per active day: **{avg_per_day:.1f}**")
                
                # Most active day analysis
                if not daily_stats.empty:
                    daily_stats['weekday'] = pd.to_datetime(daily_stats['date']).dt.day_name()
                    weekday_activity = daily_stats.groupby('weekday')['total_exercises'].sum().sort_values(ascending=False)
                    if not weekday_activity.empty:
                        most_active_day = weekday_activity.index[0]
                        st.write(f"• Most active day: **{most_active_day}**")
        
        # Recent workouts with better formatting
        st.subheader("🕐 Recent Workouts")
        if not exercises_df.empty:
            recent_exercises = exercises_df.head(10)[['date', 'exercise_name', 'sets', 'reps', 'category']]
            recent_exercises = recent_exercises.rename(columns={
                'exercise_name': 'Exercise',
                'sets': 'Sets',
                'reps': 'Reps',
                'category': 'Category',
                'date': 'Date'
            })
            
            # Style the dataframe
            st.dataframe(
                recent_exercises,
                use_container_width=True,
                height=350
            )
    
    elif tab == "Manage Exercises":
        st.header("⚙️ Manage Your Exercises")
        
        # Show recent exercises with delete option
        recent_exercises = tracker.db.get_exercises_by_date_range(
            (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
            datetime.now().strftime("%Y-%m-%d")
        )
        
        if not recent_exercises.empty:
            st.subheader("Recent Exercises (Last 7 days)")
            
            for idx, row in recent_exercises.iterrows():
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"**{row['date']}** - {row['exercise_name'].title()}: {row['sets']} sets × {row['reps']} reps")
                with col2:
                    if st.button("Delete", key=f"del_{row['id']}"):
                        tracker.db.delete_exercise(row['id'])
                        st.success("Exercise deleted!")
                        st.rerun()
        else:
            st.info("No recent exercises to manage.")

if __name__ == "__main__":
    create_dashboard()


