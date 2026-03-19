# İleride yardımcı fonksiyonlar eklemek için
def print_dataset_info(train_gen, val_gen):
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    print(f"Number of classes: {train_gen.num_classes}")