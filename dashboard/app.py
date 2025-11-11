"""
Streamlit Dashboard - Real-Time Event Visualization
Interactive dashboard for monitoring detected events and trends.
"""

import os
import sys
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from storage_manager import get_storage_manager
from event_detector import EventDetector

# Page configuration
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
