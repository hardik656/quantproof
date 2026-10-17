-- DEMO DATA ONLY: synthetic rows in the kv_state shape written by backend/db.py.
-- Not production data. Safe to re-run.
DELETE FROM kv_state WHERE key LIKE '%-demo%' OR key LIKE 'demo-%';
INSERT INTO kv_state (namespace, key, value, updated_at) VALUES
  ('report_tokens', 'RPT-demo1', 'true', now() - interval '3 days'),
  ('report_tokens', 'LITE-demo1', 'true', now() - interval '2 days'),
  ('report_tokens', 'LITE-demo2', 'true', now() - interval '1 day'),
  ('report_tokens', 'PORT-demo1', 'true', now() - interval '5 hours'),
  ('pro_tokens', 'PRO-demo-active', to_json(extract(epoch FROM now() + interval '20 days')), now() - interval '10 days'),
  ('pro_tokens', 'PRO-demo-expired', to_json(extract(epoch FROM now() - interval '2 days')), now() - interval '32 days'),
  ('tv_sessions', 'demo-session-1', json_build_object(
      'created_at', extract(epoch FROM now() - interval '1 day'),
      'expires_at', extract(epoch FROM now() + interval '6 days'),
      'alerts', json_build_array(
          json_build_object('symbol','AAPL','action','buy','price',190.1),
          json_build_object('symbol','AAPL','action','sell','price',193.4))), now() - interval '1 hour');
