-- Seeds the intersection the system runs on today, mirroring
-- src/mock/intersection.config.json so the DB and the JSON agree.
-- Re-runnable: existing rows are left untouched.

INSERT INTO intersections (intersection_id, name, lat, lon, topology, controller_name, agg_window_sec)
VALUES (
    'INT-001',
    'แยกทดสอบ INT-001',
    13.1740, 100.9270,
    '{"lanes": ["N1","N2","E1","E2","S1","S2","W1","W2"]}'::jsonb,
    'MAXPRESSURE_SWITCHING_LOSS',
    10
)
ON CONFLICT (intersection_id) DO NOTHING;

INSERT INTO intersection_phases
    (intersection_id, phase, display_order, min_green_sec, max_green_sec, yellow_sec, all_red_sec, avg_car_passed)
VALUES
    ('INT-001', 'N_GO', 1, 5, 90, 3, 2, 15),
    ('INT-001', 'E_GO', 2, 5, 90, 3, 2, 15),
    ('INT-001', 'S_GO', 3, 5, 90, 3, 2, 15),
    ('INT-001', 'W_GO', 4, 5, 90, 3, 2, 15)
ON CONFLICT (intersection_id, phase) DO NOTHING;

-- Counting camera (no video feed) plus the four CCTV feeds the dashboard plays.
INSERT INTO cameras (camera_id, intersection_id, label, direction, rtsp_url)
VALUES
    ('CAM-01', 'INT-001', 'กล้องนับรถหลัก',   NULL, NULL),
    ('cam-01', 'INT-001', 'แยกทิศเหนือ',      'N',  'rtsp://mediamtx:8554/cam-01'),
    ('cam-02', 'INT-001', 'แยกทิศตะวันออก',   'E',  'rtsp://mediamtx:8554/cam-02'),
    ('cam-03', 'INT-001', 'แยกทิศใต้',        'S',  'rtsp://mediamtx:8554/cam-03'),
    ('cam-04', 'INT-001', 'แยกทิศตะวันตก',    'W',  'rtsp://mediamtx:8554/cam-04')
ON CONFLICT (camera_id) DO NOTHING;
