# Terminal 1: Memory Test
cd /path/to/nc-diffusion
python -c "
from config import Config
from diffusion_model_optimized import build_diffusion_model_memory_efficient, monitor_memory
from utils import get_model_info

# Setup
monitor_memory()
config = Config()

# Test verschiedene Modell-Größen
print('=== Testing Model Sizes ===')
for hidden_dim in [128, 256, 512]:
    print(f'\nTesting hidden_dim={hidden_dim}')
    try:
        model = build_diffusion_model_memory_efficient(
            phi_dim=config.PHI_DIM,
            target_dim=config.TARGET_DIM,
            hidden_dim=hidden_dim
        )
        get_model_info(model)
        del model  # Speicher freigeben
        print(f'✓ hidden_dim={hidden_dim} works')
    except Exception as e:
        print(f'✗ hidden_dim={hidden_dim} failed: {e}')
"