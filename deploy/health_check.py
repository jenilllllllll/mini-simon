#!/usr/bin/env python3
"""
Health Check Endpoint for Mini-Simon
=====================================
FastAPI endpoint to verify all components are functioning correctly.

Endpoints:
  - GET /health       - Basic health check
  - GET /health/full  - Detailed system health
  - GET /ready        - Kubernetes-style readiness probe
  - GET /live         - Kubernetes-style liveness probe
"""

from __future__ import annotations

import os
import sys
import time
import psutil
from datetime import datetime
from typing import Dict, Any, Optional

import pytz

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

IST = pytz.timezone("Asia/Kolkata")


class HealthStatus(BaseModel):
    """Health check response model."""
    status: str
    timestamp: str
    uptime_seconds: float
    version: str = "1.0.0"


class FullHealthStatus(HealthStatus):
    """Detailed health check response."""
    system: Dict[str, Any]
    components: Dict[str, Any]
    environment: str


def get_system_metrics() -> Dict[str, Any]:
    """Get current system resource metrics."""
    try:
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        return {
            "cpu_percent": cpu_percent,
            "memory": {
                "total_mb": memory.total // (1024 * 1024),
                "available_mb": memory.available // (1024 * 1024),
                "percent_used": memory.percent,
            },
            "disk": {
                "total_gb": disk.total // (1024 * 1024 * 1024),
                "free_gb": disk.free // (1024 * 1024 * 1024),
                "percent_used": disk.percent,
            },
            "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else None,
        }
    except Exception as e:
        return {"error": str(e)}


def check_components() -> Dict[str, Any]:
    """Check the status of various application components."""
    components = {}
    
    # Check environment variables
    required_env = ['FYERS_APP_ID', 'FYERS_ACCESS_TOKEN']
    env_status = {var: os.getenv(var) is not None for var in required_env}
    components["environment_variables"] = {
        "all_present": all(env_status.values()),
        "details": env_status
    }
    
    # Check log directory
    log_dir = "/var/log/mini-simon"
    components["logging"] = {
        "log_directory_exists": os.path.exists(log_dir),
        "writable": os.access(log_dir, os.W_OK) if os.path.exists(log_dir) else False,
    }
    
    # Check application directories
    app_dir = "/opt/mini-simon"
    components["application"] = {
        "directory_exists": os.path.exists(app_dir),
        "venv_exists": os.path.exists(os.path.join(app_dir, "venv")),
    }
    
    return components


# Global start time for uptime calculation
START_TIME = time.time()

if HAS_FASTAPI:
    router = APIRouter(prefix="/health", tags=["health"])
    
    @router.get("", response_model=HealthStatus)
    async def health_check() -> HealthStatus:
        """Basic health check endpoint."""
        return HealthStatus(
            status="healthy",
            timestamp=datetime.now(IST).isoformat(),
            uptime_seconds=time.time() - START_TIME,
        )
    
    @router.get("/full", response_model=FullHealthStatus)
    async def full_health_check() -> FullHealthStatus:
        """Detailed health check with system metrics."""
        system_metrics = get_system_metrics()
        components = check_components()
        
        # Determine overall status
        status = "healthy"
        if not components["environment_variables"]["all_present"]:
            status = "degraded"
        if system_metrics.get("memory", {}).get("percent_used", 0) > 95:
            status = "critical"
            
        return FullHealthStatus(
            status=status,
            timestamp=datetime.now(IST).isoformat(),
            uptime_seconds=time.time() - START_TIME,
            system=system_metrics,
            components=components,
            environment=os.getenv("NODE_ENV", "production"),
        )
    
    @router.get("/ready")
    async def readiness_probe() -> Dict[str, str]:
        """Kubernetes readiness probe - checks if app is ready to serve traffic."""
        components = check_components()
        
        if not components["environment_variables"]["all_present"]:
            raise HTTPException(
                status_code=503, 
                detail="Missing required environment variables"
            )
        
        return {"status": "ready"}
    
    @router.get("/live")
    async def liveness_probe() -> Dict[str, str]:
        """Kubernetes liveness probe - checks if app is alive."""
        return {"status": "alive"}


# CLI interface for manual health checks
def run_health_check() -> None:
    """Run health check from command line."""
    print("=" * 60)
    print("Mini-Simon Health Check")
    print("=" * 60)
    
    print(f"\n📅 Time: {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"⏱️  Uptime: {time.time() - START_TIME:.2f} seconds")
    
    print("\n🖥️  System Metrics:")
    metrics = get_system_metrics()
    if "error" in metrics:
        print(f"   Error: {metrics['error']}")
    else:
        print(f"   CPU: {metrics['cpu_percent']}%")
        print(f"   Memory: {metrics['memory']['percent_used']}% " +
              f"({metrics['memory']['available_mb']}MB available)")
        print(f"   Disk: {metrics['disk']['percent_used']}% " +
              f"({metrics['disk']['free_gb']}GB free)")
    
    print("\n🔌 Components:")
    components = check_components()
    for name, status in components.items():
        icon = "✅" if status.get("all_present", status.get("exists", True)) else "❌"
        print(f"   {icon} {name}: {status}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_health_check()
