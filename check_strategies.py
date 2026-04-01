from live_engine import EngineManager
from config import get_config
import logging

logging.basicConfig(level=logging.INFO)

manager = EngineManager()
config = manager.config
print('Engine configuration:')
for key, value in config.items():
    print(f'  {key}: {type(value).__name__}')
    
print(f'Data feed symbols: {len(config.get("data_feed", {}).get("symbols", []))}')
print(f'Strategy runner config: {"strategy_runner" in config}')
print(f'Signal aggregator config: {"signal_aggregator" in config}')
