"""
Streamlit Dashboard - Interactive dashboard for monitoring detected events and trends.
"""

import os
import sys
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from storage_manager import get_storage_manager
from event_detector import EventDetector

st.set_page_config(
    page_title="Real-Time Event Detection",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .critical-alert {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .high-alert {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


def init_storage():
    """Initialize storage manager"""
    try:
        return get_storage_manager()
    except Exception as e:
        st.error(f"Failed to connect to database: {e}")
        st.info("Make sure PostgreSQL is running and configured correctly.")
        st.stop()


def load_data(storage, hours=24):
    """Load data from database"""
    try:
        events = storage.get_recent_events(hours=hours, limit=1000)
        detected = storage.get_detected_events(hours=hours)
        stats = storage.get_category_stats(hours=hours)
        return events, detected, stats
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return [], [], {}


def display_metrics(events, detected, stats):
    """Display key metrics"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 Total Events",
            value=len(events),
            delta=f"+{len([e for e in events if e['processed_at'] > datetime.utcnow() - timedelta(hours=1)])}/hr"
        )
    
    with col2:
        critical_count = len([d for d in detected if d.get('severity') == 'critical'])
        st.metric(
            label="🚨 Critical Alerts",
            value=critical_count,
            delta="Urgent" if critical_count > 0 else "Normal",
            delta_color="inverse"
        )
    
    with col3:
        high_severity = len([e for e in events if e.get('severity_score', 0) >= 0.7])
        st.metric(
            label="⚠️ High Severity",
            value=high_severity,
            delta=f"{(high_severity/max(len(events), 1)*100):.1f}%"
        )
    
    with col4:
        categories = len(stats)
        st.metric(
            label="📑 Active Categories",
            value=categories,
            delta=f"{len([s for s in stats.values() if s['count'] > 10])} trending"
        )


def display_alerts(detected):
    """Display recent alerts"""
    st.subheader("🔔 Recent Alerts")
    
    if not detected:
        st.info("No alerts in the selected time period.")
        return
    
    # Filter for important alerts
    critical = [d for d in detected if d.get('severity') == 'critical']
    high = [d for d in detected if d.get('severity') == 'high']
    
    if critical:
        st.markdown("### 🚨 Critical Alerts")
        for alert in critical[:5]:
            with st.expander(f"🔴 {alert.get('category', 'Unknown').upper()} - {alert.get('detection_type', 'Unknown')}"):
                details = alert.get('details', {})
                if isinstance(details, str):
                    details = json.loads(details)
                
                st.write(f"**Type:** {details.get('type', 'Unknown')}")
                st.write(f"**Category:** {details.get('category', 'Unknown')}")
                st.write(f"**Event Count:** {details.get('event_count', 0)}")
                st.write(f"**Detected:** {alert.get('detected_at', 'Unknown')}")
                
                if details.get('keyword'):
                    st.write(f"**Keyword:** {details.get('keyword')}")
                    st.write(f"**Spike Ratio:** {details.get('spike_ratio', 0):.2f}x")
                
                if details.get('location'):
                    st.write(f"**Location:** {details.get('location')}")
    
    if high:
        st.markdown("### ⚠️ High Priority Alerts")
        for alert in high[:5]:
            with st.expander(f"🟠 {alert.get('category', 'Unknown').upper()} - {alert.get('detection_type', 'Unknown')}"):
                details = alert.get('details', {})
                if isinstance(details, str):
                    details = json.loads(details)
                
                st.write(f"**Type:** {details.get('type', 'Unknown')}")
                st.write(f"**Event Count:** {details.get('event_count', 0)}")
                st.write(f"**Detected:** {alert.get('detected_at', 'Unknown')}")


def plot_timeline(events):
    """Plot event timeline"""
    st.subheader("📈 Event Timeline")
    
    if not events:
        st.info("No events to display.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(events)
    df['processed_at'] = pd.to_datetime(df['processed_at'])
    df['hour'] = df['processed_at'].dt.floor('H')
    
    # Group by hour and category
    hourly_counts = df.groupby(['hour', 'category']).size().reset_index(name='count')
    
    # Create timeline plot
    fig = px.line(
        hourly_counts,
        x='hour',
        y='count',
        color='category',
        title='Events Over Time by Category',
        labels={'hour': 'Time', 'count': 'Event Count', 'category': 'Category'}
    )
    
    fig.update_layout(
        height=400,
        hovermode='x unified',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_category_distribution(stats):
    """Plot category distribution"""
    st.subheader("📊 Category Distribution")
    
    if not stats:
        st.info("No category data available.")
        return
    
    # Prepare data
    categories = []
    counts = []
    avg_severities = []
    
    for category, data in stats.items():
        categories.append(category)
        counts.append(data['count'])
        avg_severities.append(data['avg_severity'] or 0)
    
    # Create subplots
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie chart
        fig = px.pie(
            values=counts,
            names=categories,
            title='Event Distribution by Category'
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Bar chart with severity
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=categories,
            y=counts,
            name='Event Count',
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title='Event Count by Category',
            xaxis_title='Category',
            yaxis_title='Count',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)


def plot_severity_heatmap(events):
    """Plot severity heatmap"""
    st.subheader("🔥 Severity Heatmap")
    
    if not events:
        st.info("No events to display.")
        return
    
    df = pd.DataFrame(events)
    df['processed_at'] = pd.to_datetime(df['processed_at'])
    df['hour'] = df['processed_at'].dt.hour
    df['date'] = df['processed_at'].dt.date
    
    # Create heatmap data
    heatmap_data = df.pivot_table(
        values='severity_score',
        index='date',
        columns='hour',
        aggfunc='mean'
    )
    
    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Hour of Day", y="Date", color="Avg Severity"),
        title="Severity Patterns by Time",
        color_continuous_scale="Reds"
    )
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)


def display_event_table(events, limit=50):
    """Display recent events table"""
    st.subheader("📋 Recent Events")
    
    if not events:
        st.info("No events to display.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(events[:limit])
    
    # Select and rename columns
    display_df = df[[
        'processed_at', 'category', 'crisis_level', 
        'severity_score', 'title', 'source'
    ]].copy()
    
    display_df.columns = ['Time', 'Category', 'Crisis Level', 'Severity', 'Title', 'Source']
    display_df['Time'] = pd.to_datetime(display_df['Time']).dt.strftime('%Y-%m-%d %H:%M')
    display_df['Severity'] = display_df['Severity'].round(2)
    
    # Apply styling
    def highlight_severity(row):
        if row['Severity'] >= 0.8:
            return ['background-color: #ffebee'] * len(row)
        elif row['Severity'] >= 0.6:
            return ['background-color: #fff3e0'] * len(row)
        else:
            return [''] * len(row)
    
    styled_df = display_df.style.apply(highlight_severity, axis=1)
    st.dataframe(styled_df, use_container_width=True, height=400)


def display_location_map(events):
    """Display world map with event locations"""
    st.subheader("🌍 Global Event Map")
    
    if not events:
        st.info("No location data available.")
        return
    
    # Extract locations
    locations = []
    for event in events:
        try:
            locs = json.loads(event.get('locations', '[]'))
            if isinstance(locs, list):
                for loc in locs:
                    locations.append({
                        'location': loc,
                        'category': event.get('category', 'unknown'),
                        'severity': event.get('severity_score', 0)
                    })
        except:
            continue
    
    if not locations:
        st.info("No location data to display.")
        return
    
    # Count by location
    location_counts = {}
    for loc_data in locations:
        loc = loc_data['location']
        if loc not in location_counts:
            location_counts[loc] = {
                'count': 0,
                'severity': 0,
                'categories': []
            }
        location_counts[loc]['count'] += 1
        location_counts[loc]['severity'] += loc_data['severity']
        location_counts[loc]['categories'].append(loc_data['category'])
    
    # Display top locations
    sorted_locs = sorted(location_counts.items(), key=lambda x: x[1]['count'], reverse=True)
    
    st.write("**Top Event Locations:**")
    for loc, data in sorted_locs[:10]:
        avg_severity = data['severity'] / data['count']
        st.write(f"📍 **{loc}**: {data['count']} events (avg severity: {avg_severity:.2f})")


def main():
    """Main dashboard"""
    
    # Header
    st.markdown('<div class="main-header">🌍 Real-Time Global Event Detection Dashboard</div>', unsafe_allow_html=True)
    st.markdown("Monitor emerging global events in real-time")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Time range selection
        time_range = st.selectbox(
            "Time Range",
            ["Last Hour", "Last 6 Hours", "Last 24 Hours", "Last 7 Days"],
            index=2
        )
        
        hours_map = {
            "Last Hour": 1,
            "Last 6 Hours": 6,
            "Last 24 Hours": 24,
            "Last 7 Days": 168
        }
        hours = hours_map[time_range]
        
        # Refresh button
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
        
        # System status
        st.markdown("---")
        st.subheader("💻 System Status")
        st.success("✅ Database Connected")
        st.info(f"📊 Monitoring last {hours} hours")
        
        # Info
        st.markdown("---")
        st.info("""
        **How it works:**
        - Data streams from NewsAPI, Reddit, GDELT
        - Real-time NLP processing
        - Event detection & clustering
        - Automated alerts via Slack/Email
        """)
    
    # Initialize storage
    storage = init_storage()

    # Load data
    with st.spinner("Loading data..."):
        events, detected, stats = load_data(storage, hours=hours)

    # Category filter in sidebar (after loading data to get available categories)
    with st.sidebar:
        st.subheader("📑 Category Filter")

        # Get all available categories
        all_categories = list(stats.keys()) if stats else []

        if all_categories:
            selected_categories = st.multiselect(
                "Select Categories to Display",
                options=all_categories,
                default=all_categories,
                help="Choose which event categories to display in the dashboard"
            )
        else:
            selected_categories = []
            st.info("No categories available")

    # Filter data based on selected categories
    if selected_categories:
        events = [e for e in events if e.get('category') in selected_categories]
        detected = [d for d in detected if d.get('category') in selected_categories]
        stats = {k: v for k, v in stats.items() if k in selected_categories}

    # Display metrics
    display_metrics(events, detected, stats)
    
    st.markdown("---")
    
    # Display alerts
    if detected:
        display_alerts(detected)
        st.markdown("---")
    
    # Visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Timeline", "📊 Categories", "🔥 Severity", "🌍 Locations"])
    
    with tab1:
        plot_timeline(events)
    
    with tab2:
        plot_category_distribution(stats)
    
    with tab3:
        plot_severity_heatmap(events)
    
    with tab4:
        display_location_map(events)
    
    st.markdown("---")
    
    # Event table
    display_event_table(events, limit=100)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        Real-Time Event Detection Pipeline | Built with Python using Streamlit, Kafka, Spark, and NLP | Ayman Siddiki
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()