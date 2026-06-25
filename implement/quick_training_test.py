# Quick test: Run 1 epoch of training

print("\n" + "="*70)
print("QUICK TRAINING TEST (1 epoch)")
print("="*70)

# Test with Lightweight3DCNN (fastest)
test_model = Lightweight3DCNN(bands=103)
dataset_path = os.path.join(root_dir, 'model', 'dataset', 'Indian_pines_corrected.mat')

try:
    history, best_loss = train_model_on_hsi(
        test_model, 
        dataset_path,
        epochs=1,  # Just 1 epoch for quick test
        batch_size=2,
        lr=1e-3,
        scale=2
    )
    
    print(f"\n✅ Training completed successfully!")
    print(f"   Loss: {best_loss:.6f}")
    print(f"   Checkpoint saved")
    print("\n🎉 TRAINING SYSTEM IS FULLY OPERATIONAL!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
