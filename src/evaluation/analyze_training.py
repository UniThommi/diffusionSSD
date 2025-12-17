# improved_training_config.py - Basierend auf Ihren Ergebnissen optimiert

class ImprovedDiffusionTrainer:
    def __init__(self, config_path="/pscratch/sd/t/tbuerger/data/optPhotonSensitiveSurface/resumFormatcurrentDistZylSSD300PMTs/resum_output_0.hdf5"):
        self.data_path = config_path
        
        # OPTIMIERTE Konfiguration basierend auf Ihren Ergebnissen
        self.config = {
            'phi_dim': 22,
            'target_dim': 7789,
            'T': 1000,
            
            # Model Architecture - VERBESSERT
            'hidden_dim': 1024,     # Beibehalten - funktioniert
            'n_layers': 4,          # Beibehalten
            't_emb_dim': 64,        # ERHÖHT von 32 - wichtig für Timestep-Encoding
            
            # Training Parameters - OPTIMIERT
            'batch_size': 32,       # Beibehalten - funktioniert
            'learning_rate': 2e-5,  # VERDOPPELT von 1e-5 - kritisch!
            'epochs': 12,           # ERHÖHT - aber mit Early Stopping
            'steps_per_epoch': 400, # REDUZIERT für schnellere Epochen
            
            # Optimization - ERWEITERT
            'gradient_clip_norm': 1.0,
            'use_ema': True,        # AKTIVIERT - sehr wichtig für Stabilität!
            'ema_decay': 0.999,
            
            # Early Stopping - NEU
            'patience': 3,          # Stoppe nach 3 Epochen ohne Verbesserung
            'min_delta': 0.01,      # Minimale Verbesserung
            
            # Learning Rate Scheduling - NEU
            'use_lr_schedule': True,
            'lr_decay_factor': 0.8,
            'lr_patience': 2,
            
            # Checkpointing
            'checkpoint_dir': './checkpoints_cpu_improved',
            'log_dir': './logs_cpu_improved',
            'save_every_n_epochs': 2,
            'eval_every_n_epochs': 3,
        }
        
        # Setup directories
        os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
        os.makedirs(self.config['log_dir'], exist_ok=True)
        
        # Setup optimizations
        self.setup_cpu_optimizations()
        
        # Early stopping
        self.early_stopping = EarlyStoppingCallback(
            patience=self.config['patience'],
            min_delta=self.config['min_delta']
        )
        
        # LR Scheduler
        self.lr_scheduler = LearningRateScheduler(
            initial_lr=self.config['learning_rate'],
            decay_factor=self.config['lr_decay_factor'],
            patience=self.config['lr_patience']
        )

class EarlyStoppingCallback:
    def __init__(self, patience=3, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.wait = 0
        self.stopped_epoch = 0
        
    def on_epoch_end(self, epoch, current_loss):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.wait = 0
            print(f"    -> Neue Best Loss: {self.best_loss:.6f}")
        else:
            self.wait += 1
            print(f"    -> Keine Verbesserung, Wait: {self.wait}/{self.patience}")
            
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            print(f"    -> Early Stopping triggered at epoch {epoch}")
            return True
        return False

class LearningRateScheduler:
    def __init__(self, initial_lr=2e-5, decay_factor=0.8, patience=2):
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.decay_factor = decay_factor
        self.patience = patience
        self.wait = 0
        self.best_loss = float('inf')
        
    def on_epoch_end(self, current_loss):
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            old_lr = self.current_lr
            self.current_lr *= self.decay_factor
            self.wait = 0
            print(f"    -> Learning Rate reduziert: {old_lr:.2e} -> {self.current_lr:.2e}")
            return self.current_lr
        
        return None

# Erweiterte Training-Methode mit EMA
def setup_training_with_ema(self, model):
    """Training Setup mit EMA"""
    # Optimizer
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=self.config['learning_rate'],
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7
    )
    
    # Loss function
    loss_fn = tf.keras.losses.MeanSquaredError()
    
    # EMA Model - WICHTIG!
    if self.config['use_ema']:
        ema_model = tf.keras.models.clone_model(model)
        ema_model.set_weights(model.get_weights())
        print("✓ EMA Model erstellt")
    else:
        ema_model = None
    
    # Metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    
    return optimizer, loss_fn, ema_model, train_loss

def update_ema(self, model, ema_model):
    """EMA Update - kritisch für Stabilität"""
    if ema_model is None:
        return
    
    decay = self.config['ema_decay']
    for ema_param, param in zip(ema_model.trainable_variables, model.trainable_variables):
        ema_param.assign(decay * ema_param + (1 - decay) * param)

# Verbesserte Training Loop
def train_epoch_improved(self, model, optimizer, loss_fn, ema_model, train_loss, dataset, epoch):
    """Verbesserte Training Epoch mit EMA"""
    print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
    start_time = time.time()
    
    # Reset metrics
    train_loss.reset_state()
    
    step_count = 0
    try:
        for batch in dataset.take(self.config['steps_per_epoch']):
            # Training step
            loss = self.train_step(model, optimizer, loss_fn, batch)
            train_loss.update_state(loss)
            
            # EMA Update - NACH jedem Step!
            if ema_model is not None:
                self.update_ema(model, ema_model)
            
            step_count += 1
            
            # Progress
            if step_count % 50 == 0:
                current_lr = optimizer.learning_rate.numpy()
                print(f"  Step {step_count}: Loss = {loss.numpy():.6f}, LR = {current_lr:.2e}")
                
    except Exception as e:
        print(f"Error in training step {step_count}: {e}")
        return train_loss.result()
    
    epoch_time = time.time() - start_time
    avg_loss = train_loss.result()
    
    print(f"  Epoch completed in {epoch_time:.1f}s")
    print(f"  Average Loss: {avg_loss:.6f}")
    
    # Early Stopping Check
    should_stop = self.early_stopping.on_epoch_end(epoch, avg_loss)
    
    # Learning Rate Scheduling
    new_lr = self.lr_scheduler.on_epoch_end(avg_loss)
    if new_lr is not None:
        optimizer.learning_rate.assign(new_lr)
    
    return avg_loss, should_stop

# Vorhersage: Mit diesen Verbesserungen sollten Sie erreichen:
# - Loss < 0.5 nach 3-4 Epochen
# - Bessere Stabilität durch EMA
# - Automatisches Stoppen bei Konvergenz
# - Adaptive Learning Rate

print("=== VERBESSERTE KONFIGURATION ===")
print("Wichtigste Änderungen:")
print("1. Learning Rate: 1e-5 -> 2e-5 (KRITISCH)")
print("2. EMA aktiviert (STABILITÄT)")
print("3. Early Stopping (EFFIZIENZ)")
print("4. t_emb_dim: 32 -> 64 (BESSERE TIMESTEP-ENCODING)")
print("5. LR Scheduling (ADAPTIVE)")
print("\nErwartete Verbesserung: Loss von ~1.0 auf ~0.3-0.5")