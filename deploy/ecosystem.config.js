/**
 * PM2 Ecosystem Configuration for Mini-Simon
 * 
 * This configuration ensures:
 * - Auto-restart on crashes
 * - Auto-start on server boot
 * - Memory limit management
 * - Log rotation
 * 
 * Usage:
 *   pm2 start ecosystem.config.js
 *   pm2 save
 *   pm2 startup systemd
 */

module.exports = {
  apps: [
    {
      name: 'mini-simon-dashboard',
      script: 'run_production.py',
      cwd: '/opt/mini-simon',
      
      // Execution settings
      interpreter: '/opt/mini-simon/venv/bin/python3',
      exec_mode: 'fork',
      instances: 1,
      
      // Environment variables
      env: {
        NODE_ENV: 'production',
        PYTHONUNBUFFERED: '1',
        PYTHONUTF8: '1',
        PYTHONIOENCODING: 'utf-8',
        FYERS_APP_ID: process.env.FYERS_APP_ID,
        FYERS_ACCESS_TOKEN: process.env.FYERS_ACCESS_TOKEN,
        DISCORD_WEBHOOK_URL: process.env.DISCORD_WEBHOOK_URL,
      },
      
      // Restart behavior
      autorestart: true,
      max_restarts: 10,
      min_uptime: '10s',
      restart_delay: 5000,
      
      // Memory management
      max_memory_restart: '2G',
      
      // Logging
      log_file: '/var/log/mini-simon/combined.log',
      out_file: '/var/log/mini-simon/out.log',
      error_file: '/var/log/mini-simon/error.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
      merge_logs: true,
      
      // Monitoring
      watch: false,
      ignore_watch: ['node_modules', 'logs', '.git', '__pycache__', '*.pyc'],
      
      // Advanced settings for uptime
      kill_timeout: 5000,
      listen_timeout: 10000,
      shutdown_with_message: true,
      
      // Health check (optional - requires pm2-health plugin)
      // health_check_fatal_exceptions: true,
    },
    {
      name: 'mini-simon-websocket-manager',
      script: 'websocket_manager_daemon.py',
      cwd: '/opt/mini-simon',
      
      interpreter: '/opt/mini-simon/venv/bin/python3',
      exec_mode: 'fork',
      instances: 1,
      
      env: {
        NODE_ENV: 'production',
        PYTHONUNBUFFERED: '1',
        PYTHONUTF8: '1',
      },
      
      autorestart: true,
      max_restarts: 15,
      min_uptime: '5s',
      restart_delay: 3000,
      
      max_memory_restart: '1G',
      
      log_file: '/var/log/mini-simon/ws-combined.log',
      out_file: '/var/log/mini-simon/ws-out.log',
      error_file: '/var/log/mini-simon/ws-error.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss Z',
      merge_logs: true,
      
      watch: false,
      kill_timeout: 3000,
    }
  ],
  
  // Deployment configuration (optional - for multi-server setups)
  deploy: {
    production: {
      user: 'mini-simon',
      host: ['your-droplet-ip'],
      ref: 'origin/main',
      repo: 'https://github.com/yourusername/mini-simon.git',
      path: '/opt/mini-simon',
      'post-deploy': 'pip install -r requirements.txt && pm2 reload ecosystem.config.js --env production',
    }
  }
};
