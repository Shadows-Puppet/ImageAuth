from data import UniversalFakeDetectDataset, TestingDataset, get_test_models

print("="*60)
print("TESTING DATASET LOADING")
print("="*60)

# Test training dataset
print("\n1. Loading training data...")
train_dataset = UniversalFakeDetectDataset(
    root_dir='../data/training',
    max_samples_per_class=10  # Only 10 per class for quick test
)

print(f"\nDataset size: {len(train_dataset)}")
print(f"First sample: {train_dataset[0]}")
print(f"Label distribution: {sum(train_dataset.labels)} fake, {len(train_dataset.labels) - sum(train_dataset.labels)} real")

# Test a few samples
print("\nSample images:")
for i in range(min(3, len(train_dataset))):
    path, label = train_dataset[i]
    label_str = "FAKE" if label == 1 else "REAL"
    print(f"  [{label_str}] {path}")

# Test testing dataset
print("\n" + "="*60)
print("2. Loading testing data...")

test_models = get_test_models('../data/testing')
print(f"Available test models: {test_models}")

if test_models:
    # Test on first model
    test_model = test_models[0]
    print(f"\nLoading test data for: {test_model}")
    
    test_dataset = TestingDataset(
        root_dir='../data/testing',
        model_name=test_model,
        max_samples=20
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    print(f"First test sample: {test_dataset[0]}")

print("\n" + "="*60)
print("DATASET TEST COMPLETE!")
print("="*60)