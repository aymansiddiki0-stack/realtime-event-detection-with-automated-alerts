-- Initialize database schema for Event Detection Pipeline

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Events table
CREATE TABLE IF NOT EXISTS events (
    id SERIAL PRIMARY KEY,
    event_id VARCHAR(255) UNIQUE NOT NULL,
    source VARCHAR(50),
    source_type VARCHAR(50),
    title TEXT,
    description TEXT,
    content TEXT,
    url TEXT,
    published_at TIMESTAMP,
    timestamp TIMESTAMP,
    category VARCHAR(100),
    category_confidence FLOAT,
    crisis_level VARCHAR(20),
    severity_score FLOAT,
    persons TEXT,
    organizations TEXT,
    locations TEXT,
    word_count INTEGER,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_category ON events(category);
CREATE INDEX IF NOT EXISTS idx_events_crisis_level ON events(crisis_level);
CREATE INDEX IF NOT EXISTS idx_events_processed_at ON events(processed_at);
CREATE INDEX IF NOT EXISTS idx_events_severity ON events(severity_score DESC);
CREATE INDEX IF NOT EXISTS idx_events_source ON events(source);

-- Detected events table
CREATE TABLE IF NOT EXISTS detected_events (
    id SERIAL PRIMARY KEY,
    detection_type VARCHAR(50),
    category VARCHAR(100),
    severity VARCHAR(20),
    event_count INTEGER,
    details JSONB,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    alert_sent BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_detected_events_severity ON detected_events(severity);
CREATE INDEX IF NOT EXISTS idx_detected_events_detected_at ON detected_events(detected_at);
CREATE INDEX IF NOT EXISTS idx_detected_events_type ON detected_events(detection_type);

-- Alerts table
CREATE TABLE IF NOT EXISTS alerts (
    id SERIAL PRIMARY KEY,
    detection_id INTEGER REFERENCES detected_events(id),
    alert_type VARCHAR(50),
    message TEXT,
    sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20)
);

CREATE INDEX IF NOT EXISTS idx_alerts_sent_at ON alerts(sent_at);
CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);

-- Pipeline metrics table
CREATE TABLE IF NOT EXISTS pipeline_metrics (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(100),
    metric_value FLOAT,
    metric_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON pipeline_metrics(metric_timestamp);
CREATE INDEX IF NOT EXISTS idx_metrics_name ON pipeline_metrics(metric_name);

-- Create views for analytics
CREATE OR REPLACE VIEW recent_high_severity AS
SELECT *
FROM events
WHERE severity_score >= 0.7
  AND processed_at > NOW() - INTERVAL '24 hours'
ORDER BY severity_score DESC, processed_at DESC;

CREATE OR REPLACE VIEW category_summary AS
SELECT 
    category,
    COUNT(*) as event_count,
    AVG(severity_score) as avg_severity,
    MAX(severity_score) as max_severity,
    COUNT(DISTINCT DATE(processed_at)) as days_active
FROM events
WHERE processed_at > NOW() - INTERVAL '7 days'
GROUP BY category
ORDER BY event_count DESC;

CREATE OR REPLACE VIEW hourly_event_rate AS
SELECT 
    DATE_TRUNC('hour', processed_at) as hour,
    category,
    COUNT(*) as event_count
FROM events
WHERE processed_at > NOW() - INTERVAL '24 hours'
GROUP BY DATE_TRUNC('hour', processed_at), category
ORDER BY hour DESC, event_count DESC;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO eventpipeline;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO eventpipeline;